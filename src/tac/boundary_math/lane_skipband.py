# SPDX-License-Identifier: MIT
"""Lane stride-2 SKIP-BAND primitives (ARM-C #524; SPEC_v10 §13.1 row 4) — numpy-fp32 reference.

DERIVATION (labeled DERIVED; chain from MEASURED facts in
``.omx/research/segnet_recursive_fractal_factorization_20260715.md`` §5):

1. MEASURED: the frozen SegNet's final stride-1 decoder block has NO skip (conv1_in=32), so
   NOTHING in the net sees full-res pixels after the stem; ALL sub-stride-4 boundary
   localization flows through the ONE stride-2 skip — 16 channels at (192, 256).
2. MEASURED: destroying sub-stride-4 detail in that skip via a 2x-down->2x-up ablation at the
   skip resolution induces 8,072 argmax flips of which Road-Lane = 77% — Lane boundary
   precision rides on the stride-2 skip far out of proportion (Lane is THE skip-limited pair).
3. DERIVED: the render-side structure that SURVIVES the skip path is therefore exactly the
   DETAIL BAND the ablation operator destroys: ``SB(x) = D2(x) - U2(D2(D2(x)))`` where
   ``D2`` is a 2x2 average-pool and ``U2`` a 2x nearest upsample — i.e. the component of the
   half-resolution image that is NOT representable at quarter resolution (the
   [stride-4 Nyquist, stride-2 Nyquist) band, measured at the skip's own (192,256) grid).
   APPROXIMATION (stated): the stem is a LEARNED 3x3 stride-2 conv, not an ideal
   band-splitter; SB is the render-side band-limited sufficient statistic UP TO that local
   linear filter. The ablation in (2) that produced the 77% number is exactly the
   complement-projection of SB, so SB is the measured carrier, not a guess.
4. DERIVED lever: supervise the WITNESS render's SB against the GT frame's SB on the LANE
   BAND (GT Lane class dilated by a small radius — Lane markings are thin all-boundary
   double-edges, 19% of d_seg flips) so the witness spends its capacity making the Lane band
   legible to the ONLY channel through which the frozen scorer localizes it.

All functions are pure numpy fp32 (the deterministic reference authority); the MLX twin in
``experiments/train_levelset_witness_realized_through_R_mlx.py`` (the ``lane_skipband`` loss
branch) must match it (parity-tested in
``src/tac/tests/test_lane_skipband.py``). Luma is BT.601 (the same coefficients the witness
chroma path and the chroma-boundary lever use). Inputs are [0,255] RGB; ``luma_bt601``
normalizes to [0,1] so the loss term is O(1)-comparable with the sibling lever terms.

research_only-adjacent: this module carries NO score claim; the lever is default-OFF and its
d_seg effect is RUN-GATED (duty-to-measure A/B). Pointer UNMOVED (means).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "luma_bt601",
    "avg_pool2",
    "up2_nearest",
    "skip_band_detail",
    "lane_band_mask_half",
    "skipband_target_and_mask",
    "skipband_term_np",
    "skipband_term_grad_np",
]


def luma_bt601(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma of an (H, W, 3) [0,255] RGB frame, normalized to [0,1] fp32."""
    a = np.asarray(rgb, dtype=np.float32)
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3) RGB, got shape {a.shape}")
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return (lum / 255.0).astype(np.float32)


def avg_pool2(x: np.ndarray) -> np.ndarray:
    """2x2 average pool of an (H, W) map (H, W must be even)."""
    a = np.asarray(x, dtype=np.float32)
    h, w = a.shape
    if h % 2 or w % 2:
        raise ValueError(f"avg_pool2 needs even dims, got {a.shape}")
    return a.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)).astype(np.float32)


def up2_nearest(x: np.ndarray) -> np.ndarray:
    """2x nearest-neighbor upsample of an (H, W) map."""
    a = np.asarray(x, dtype=np.float32)
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)


def skip_band_detail(lum: np.ndarray) -> np.ndarray:
    """The stride-2 skip DETAIL BAND of a full-res (H, W) luma map, at (H/2, W/2).

    ``SB = D2(lum) - U2(D2(D2(lum)))`` — the component of the half-res image destroyed by the
    fractal memo's skip-ablation operator (2x down->up at the skip resolution). This is the
    render-side band that carries the MEASURED 77% of Lane skip-detail flips (§5).
    """
    x2 = avg_pool2(lum)
    x4 = avg_pool2(x2)
    return (x2 - up2_nearest(x4)).astype(np.float32)


def _dilate_chebyshev(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation with a (2r+1)^2 square structuring element (Chebyshev ball), numpy-only."""
    m = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return m.copy()
    out = m.copy()
    for _ in range(radius):
        p = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
            | p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
        )
    return out


def lane_band_mask_half(lstar: np.ndarray, dilate: int = 2, lane_class: int = 1) -> np.ndarray:
    """Lane-band {0,1} mask at HALF resolution from a full-res (H, W) GT argmax map.

    Lane (class 1, comma10k CANONICAL order — CLAUDE.md NON-NEGOTIABLE; do NOT luma-sort)
    pixels dilated by ``dilate`` (Chebyshev) at full res, then max-pooled 2x (a half-res cell
    is in-band iff ANY of its 4 pixels is). Lane markings are thin all-boundary double-edges,
    so the dilated class support IS the boundary band.
    """
    ls = np.asarray(lstar)
    lane = _dilate_chebyshev(ls == int(lane_class), int(dilate))
    h, w = lane.shape
    if h % 2 or w % 2:
        raise ValueError(f"lane_band_mask_half needs even dims, got {lane.shape}")
    half = lane.reshape(h // 2, 2, w // 2, 2).max(axis=(1, 3))
    return half.astype(np.float32)


def skipband_target_and_mask(
    gt_rgb_seg: np.ndarray, lstar: np.ndarray, *, dilate: int = 2, lane_class: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair theta-independent constants: (SB target from the GT frame, lane-band mask).

    ``gt_rgb_seg`` is the GT frame AS SegNet reads it — (SEG_H, SEG_W, 3) [0,255] (bilinear
    resize of the camera frame, align_corners=False, same as ``SegNet.preprocess_input``).
    """
    sb = skip_band_detail(luma_bt601(gt_rgb_seg))
    mask = lane_band_mask_half(lstar, dilate=dilate, lane_class=lane_class)
    if sb.shape != mask.shape:
        raise ValueError(f"SB {sb.shape} vs mask {mask.shape} shape mismatch")
    return sb, mask


def skipband_term_np(render_rgb_seg: np.ndarray, sb_gt: np.ndarray, mask: np.ndarray) -> float:
    """Reference loss term: masked mean squared SB error, matching the MLX branch bit-for-bit
    in exact arithmetic: ``sum(mask * (SB(render) - sb_gt)^2) / (sum(mask) + 1e-6)``."""
    sb_w = skip_band_detail(luma_bt601(render_rgb_seg))
    m = np.asarray(mask, dtype=np.float32)
    num = float(np.sum(m * np.square(sb_w - np.asarray(sb_gt, np.float32))))
    return num / (float(np.sum(m)) + 1e-6)


def skipband_term_grad_np(
    render_rgb_seg: np.ndarray, sb_gt: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """CLOSED-FORM gradient of ``skipband_term_np`` w.r.t. the full-res render LUMA (H, W).

    The term is a quadratic in the render through the LINEAR operator ``SB ∘ luma``:
        d/dlum = A^T ( 2 * mask * (SB(lum) - sb_gt) / (sum(mask)+1e-6) )
    where ``A = (I - U2 D2) D2`` acting on luma. ``A^T = D2^T (I - D2^T U2^T)`` with
    ``D2^T = up2_nearest * 0.25`` (adjoint of 2x2 mean) and ``U2^T = 4 * avg_pool2 * ...``
    — concretely ``U2^T y = sum over the 2x2 cell = 4*avg_pool2(y)``. Used by the $0
    bindingness probe (nonzero-gradient proof on the REAL render); the trainer path uses MLX
    autodiff on the identical expression.
    """
    lum = luma_bt601(render_rgb_seg)
    sb_w = skip_band_detail(lum)
    m = np.asarray(mask, dtype=np.float32)
    g_half = 2.0 * m * (sb_w - np.asarray(sb_gt, np.float32)) / (float(np.sum(m)) + 1e-6)
    # adjoint of (I - U2 D2) at half res: y - D2^T-free part; U2^T y = 4*avg_pool2(y)... careful:
    # (U2 D2)^T = D2^T U2^T. U2^T y (quarter res) = sum of each 2x2 cell of y = 4*avg_pool2(y).
    # D2^T z (half res from quarter) = up2_nearest(z) * 0.25.
    u2t = 4.0 * avg_pool2(g_half)               # (H/4, W/4)
    inner = g_half - 0.25 * up2_nearest(u2t)    # (I - D2^T U2^T) g   at half res
    # outer D2^T back to full res:
    g_full = 0.25 * up2_nearest(inner)          # (H, W) gradient w.r.t. luma in [0,1]
    return g_full.astype(np.float32)
