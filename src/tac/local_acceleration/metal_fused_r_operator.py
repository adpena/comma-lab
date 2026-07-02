# SPDX-License-Identifier: MIT
"""Fused contest-faithful R-operator (eval roundtrip) as a custom MLX op.

What the R operator is
----------------------
The witness's realized-d_seg path renders RGB at sub-camera resolution and then
runs the **contest-EXACT eval roundtrip** before the frozen scorer reads it:

    render (Hin x Win) --bicubic up--> CAMERA (874 x 1164)
                       --uint8 STE @ CAMERA-->  (the stored video quantization)
                       --bilinear down--> SCORER (384 x 512)   [float, no trailing uint8]

This is exactly ``tac.local_acceleration.pr95_hnerv_mlx_training.apply_contest_faithful_roundtrip_nhwc``
(the witness production R; see ``upstream/evaluate.py`` + ``upstream/modules.py:108-113``).
That MLX function is the **oracle** this module matches bit-for-bit. The R
roundtrip runs **twice per pair** (f0, f1) inside the loss, and MLX's separable
per-axis resize materializes a large ``(B, out, taps, W, C)`` intermediate per
axis (5 ops + a round), which is the clearest *forward* gap vs torch-MPS
(`mlx_vs_torch_mps_bench` §2/§5: CPU render_R 0.07-0.11x). This module collapses
the **5 separable MLX passes + the round into 2 on-device Metal kernels** (one
fused bicubic-up+clip+round @ camera, one fused bilinear-down to scorer), with no
intermediate Python/numpy recovery.

Authority note (NO-FAKE)
------------------------
This is a GRADIENT/COMPUTE throughput tool on the MLX (Metal) substrate. It is
**never** a d_seg/d_pose score authority — the FP32-exact numpy/torch-CPU scorer
on the exact byte-closed archive remains the only score authority. A faster R is
not a better score. Per CLAUDE.md "Native eval-time runtime discipline": this
ships a numpy reference oracle (``fused_r_forward_numpy`` / ``fused_r_vjp_numpy``)
and a bit-identical / sub-LSB-documented equivalence harness against the MLX
production R.

Coefficient note (matches the REAL oracle, not the loose memo wording)
----------------------------------------------------------------------
The MLX production R uses the **Catmull/PyTorch-default cubic coefficient
``a = -0.75``** (``_cubic_convolution_weight`` in ``pr95_hnerv_mlx_training``),
NOT ``a = -0.5`` as the campaign memo/prompt loosely state. This module matches
the real code (``a = -0.75``) so the kernel is bit-faithful to the witness's
production R. ``mx.round`` is round-half-to-even (verified on this host: matches
``np.rint``); the Metal kernel uses ``rint()`` (round-to-nearest-even).

Design (forward = fast metal; backward = fully-fused metal transpose VJP)
------------------------------------------------------------------------
* FORWARD: two ``mx.fast.metal_kernel`` launches (up+clip+round, then down),
  W-outer / H-inner tap order == MLX's H-pass-then-W-pass order => bit-faithful.
* VJP (P2b, 2026-07-02): a fully-fused **metal transpose VJP** —
  ``grad_x = Up^T( clip_mask * Down^T( cotangent ) )`` computed by four launches of a
  generic transpose-resample kernel (one thread per INPUT pixel gather-sums its
  precomputed transpose taps in ascending forward-output order; NO atomics ->
  DETERMINISTIC). This REPLACES the prior ``mx.vjp`` of the pure-MLX oracle, which was
  correct but re-ran the slow MLX forward and carried a ~1-ULP scatter
  non-determinism floor. The transpose tap tables are the inverse index of the
  forward taps, with same-``(out,in)`` clipped taps PRE-COMBINED (the exact adjoint
  weight; avoids edge double-counting). The kernel is BIT-IDENTICAL (atol=0) to the
  deterministic numpy authority ``fused_r_vjp_numpy`` — which was itself made
  deterministic (explicit ascending-output accumulation instead of BLAS-``einsum``,
  whose blocked-sgemm order is not metal-reproducible; see ``_transpose_sorted_taps``).
  The numpy ``fused_r_vjp_numpy`` is the analytic-transpose authority; the legacy
  BLAS path is preserved as ``_fused_r_vjp_numpy_einsum_legacy`` for the residual
  report + regression guard.

Contract (NHWC, leading dims preserved):
- input  : ``(..., Hin, Win, 3)`` float RGB in [0, 255]
- output : ``(..., output_hw[0], output_hw[1], 3)`` float (scorer-res, no uint8)
"""

from __future__ import annotations

from typing import Any

import numpy as np

CAMERA_HW: tuple[int, int] = (874, 1164)
SCORER_HW: tuple[int, int] = (384, 512)
CUBIC_A: float = -0.75  # PyTorch-default / MLX production R coefficient.

__all__ = [
    "CAMERA_HW",
    "SCORER_HW",
    "CUBIC_A",
    "fused_r_forward_numpy",
    "fused_r_vjp_numpy",
    "resize_indices_weights_numpy",
    "metal_fused_r_available",
    "make_fused_r_roundtrip",
    "fused_r_roundtrip",
    "fused_r_roundtrip_reference",
    "assert_metal_matches_cpu_oracle",
    "assert_metal_vjp_deterministic",
]


# --------------------------------------------------------------------------- #
# NUMPY ORACLE — bit-faithful mirror of the MLX production R resize math.
# --------------------------------------------------------------------------- #


def _cubic_weight_numpy(distance: np.ndarray, a: float = CUBIC_A) -> np.ndarray:
    """Mirror of ``pr95_hnerv_mlx_training._cubic_convolution_weight`` (a=-0.75)."""

    x = np.abs(distance).astype(np.float32)
    x2 = x * x
    x3 = x2 * x
    inner = (a + 2.0) * x3 - (a + 3.0) * x2 + 1.0
    outer = a * x3 - 5.0 * a * x2 + 8.0 * a * x - 4.0 * a
    return np.where(x <= 1.0, inner, np.where(x < 2.0, outer, 0.0)).astype(np.float32)


def resize_indices_weights_numpy(
    *, in_size: int, out_size: int, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror of ``pr95_hnerv_mlx_training._resize_indices_weights`` (align_corners=False).

    Returns ``(indices int32 (out_size, taps), weights float32 (out_size, taps))``
    with indices clipped to ``[0, in_size - 1]`` (weights NOT renormalized — exactly
    as the MLX code does it).
    """

    in_size = int(in_size)
    out_size = int(out_size)
    scale = np.float32(in_size) / np.float32(out_size)
    out = np.arange(out_size, dtype=np.float32)
    real = (out + np.float32(0.5)) * scale - np.float32(0.5)
    base = np.floor(real).astype(np.int32)
    mode = mode.strip().lower()
    if mode == "bilinear":
        indices = np.stack([base, base + 1], axis=1)
        right = (real - base.astype(np.float32)).astype(np.float32)
        weights = np.stack([np.float32(1.0) - right, right], axis=1).astype(np.float32)
    elif mode == "bicubic":
        offsets = np.array([-1, 0, 1, 2], dtype=np.int32)
        indices = base[:, None] + offsets[None, :]
        weights = _cubic_weight_numpy(real[:, None] - indices.astype(np.float32))
    else:
        raise ValueError(f"unsupported resize mode: {mode}")
    indices = np.clip(indices, 0, in_size - 1).astype(np.int32)
    return indices, weights.astype(np.float32)


def _resample_pair_numpy(
    x_bhwc: np.ndarray,
    *,
    hidx: np.ndarray,
    hw: np.ndarray,
    widx: np.ndarray,
    ww: np.ndarray,
    do_round: bool,
) -> np.ndarray:
    """Separable 2D resample (H-pass then W-pass) on (B,Hin,Win,C) float32.

    Order matches MLX exactly: H-axis resize fully, then W-axis resize. With
    ``do_round`` the result is clip[0,255] then round-half-to-even (the camera
    uint8 STE forward value). Pure-numpy, float32 throughout.
    """

    x = np.asarray(x_bhwc, dtype=np.float32)
    B, Hin, Win, C = x.shape
    out_h, taps_h = hidx.shape
    out_w, taps_w = widx.shape
    # H-pass: gather rows then weighted sum over taps -> (B, out_h, Win, C)
    # x[:, hidx, :, :] -> (B, out_h, taps_h, Win, C); weights (1,out_h,taps_h,1,1)
    gathered_h = x[:, hidx, :, :]  # (B, out_h, taps_h, Win, C)
    y = np.sum(
        gathered_h * hw[None, :, :, None, None].astype(np.float32), axis=2
    ).astype(np.float32)  # (B, out_h, Win, C)
    # W-pass: gather cols then weighted sum -> (B, out_h, out_w, C)
    gathered_w = y[:, :, widx, :]  # (B, out_h, out_w, taps_w, C)
    out = np.sum(
        gathered_w * ww[None, None, :, :, None].astype(np.float32), axis=3
    ).astype(np.float32)  # (B, out_h, out_w, C)
    if do_round:
        out = np.clip(out, 0.0, 255.0).astype(np.float32)
        out = np.rint(out).astype(np.float32)  # round-half-to-even == mx.round
    return out


def fused_r_forward_numpy(
    x_nhwc: np.ndarray,
    *,
    camera_hw: tuple[int, int] = CAMERA_HW,
    output_hw: tuple[int, int] = SCORER_HW,
    ste_round: bool = True,
) -> np.ndarray:
    """Numpy oracle for the contest-faithful R: up(bicubic)->uint8@camera->down(bilinear).

    Bit-faithful mirror of
    ``pr95_hnerv_mlx_training.apply_contest_faithful_roundtrip_nhwc``. Leading dims
    are preserved.
    """

    x = np.asarray(x_nhwc, dtype=np.float32)
    if x.ndim < 4 or x.shape[-1] != 3:
        raise ValueError(f"expected (..., H, W, 3) RGB, got {x.shape}")
    lead = x.shape[:-3]
    Hin, Win = int(x.shape[-3]), int(x.shape[-2])
    flat = x.reshape((-1, Hin, Win, 3))
    cam_h, cam_w = int(camera_hw[0]), int(camera_hw[1])
    out_h, out_w = int(output_hw[0]), int(output_hw[1])
    up_h_idx, up_h_w = resize_indices_weights_numpy(in_size=Hin, out_size=cam_h, mode="bicubic")
    up_w_idx, up_w_w = resize_indices_weights_numpy(in_size=Win, out_size=cam_w, mode="bicubic")
    # Up + (clip+round) @ camera res.
    camera = _resample_pair_numpy(
        flat, hidx=up_h_idx, hw=up_h_w, widx=up_w_idx, ww=up_w_w, do_round=bool(ste_round)
    )
    if not ste_round:
        camera = np.clip(camera, 0.0, 255.0).astype(np.float32)
    dn_h_idx, dn_h_w = resize_indices_weights_numpy(in_size=cam_h, out_size=out_h, mode="bilinear")
    dn_w_idx, dn_w_w = resize_indices_weights_numpy(in_size=cam_w, out_size=out_w, mode="bilinear")
    out = _resample_pair_numpy(
        camera, hidx=dn_h_idx, hw=dn_h_w, widx=dn_w_idx, ww=dn_w_w, do_round=False
    )
    return out.reshape((*lead, out_h, out_w, 3)).astype(np.float32)


# --------------------------------------------------------------------------- #
# NUMPY VJP ORACLE — analytic transpose of the two linear resamples + clip mask.
# --------------------------------------------------------------------------- #


def _axis_matrix_numpy(*, in_size: int, out_size: int, mode: str) -> np.ndarray:
    """Dense (out_size, in_size) float32 resample matrix (sum of clipped taps).

    Kept for the legacy (BLAS-``einsum``) VJP path used only by the residual/regression
    guard ``_fused_r_vjp_numpy_einsum_legacy`` — the live ``fused_r_vjp_numpy`` no
    longer routes through a dense matrix (see the deterministic-transpose note below).
    """

    idx, w = resize_indices_weights_numpy(in_size=in_size, out_size=out_size, mode=mode)
    out_size = int(out_size)
    in_size = int(in_size)
    mat = np.zeros((out_size, in_size), dtype=np.float32)
    for o in range(out_size):
        for t in range(idx.shape[1]):
            mat[o, int(idx[o, t])] += np.float32(w[o, t])
    return mat


# --------------------------------------------------------------------------- #
# DETERMINISTIC TRANSPOSE TABLES + AXIS OP (the Metal-parity numpy authority).
#
# WHY (measured 2026-07-02, P2b): the prior ``fused_r_vjp_numpy`` computed each
# separable axis-transpose via ``np.einsum(..., optimize=True)`` over a dense
# resample matrix. With numpy on Accelerate BLAS, that einsum routes to a BLOCKED
# ``sgemm`` whose float32 accumulation ORDER is an implementation artifact (NOT a
# spec) and is NOT reproducible in a Metal kernel: measured only ~34% of the Up^T
# outputs bit-match a sequential accumulation (max|Δ|~1.9e-6). To make the numpy
# authority a DETERMINISTIC target the metal transpose VJP can match BIT-FOR-BIT
# (CLAUDE.md deterministic-reproducibility principle), the transpose is computed in
# an explicit, well-defined accumulation order: for each INPUT pixel, sum its taps
# in INCREASING forward-output-index order with float32 multiply-then-add. This is
# the exact same order the Metal kernel uses (one thread per input pixel, ``#pragma
# clang fp contract(off)``), so metal == numpy at atol=0. The result remains a valid
# analytic VJP: it is the exact adjoint of the linear resamples (taps that clip to
# the same (out,in) pair are pre-COMBINED, which is the true ∂y[o]/∂x[i]), and it
# still agrees with ``mx.vjp`` autodiff to float tol (the ordering delta ~1e-5 «
# the 2e-4 gate). ``_fused_r_vjp_numpy_einsum_legacy`` preserves the old BLAS path
# for the transparent residual report + a regression guard.
# --------------------------------------------------------------------------- #

_transpose_np_cache: dict[str, Any] = {}


def _transpose_sorted_taps(*, in_size: int, out_size: int, mode: str) -> dict[str, Any]:
    """Transpose of the forward (in->out) resample, as per-input sorted taps.

    The forward maps input ``i`` to outputs via ``resize_indices_weights_numpy``.
    The transpose maps output cotangents back to inputs: for each INPUT pixel ``i``
    the taps are the set of ``(o, w_combined)`` where forward output ``o`` reads
    input ``i`` (taps clipping to the same ``(o, i)`` pre-COMBINED — the exact
    adjoint weight ``∂y[o]/∂x[i]``), SORTED by ``o`` ascending. Returns both a padded
    ``(in_size, maxt)`` form (for the vectorized numpy reference) and a flat CSR form
    (``starts``/``counts``/``flat_idx``/``flat_w`` for the Metal kernel). Cached.
    """

    in_size = int(in_size)
    out_size = int(out_size)
    key = f"{in_size}:{out_size}:{mode}"
    cached = _transpose_np_cache.get(key)
    if cached is not None:
        return cached
    idx, w = resize_indices_weights_numpy(in_size=in_size, out_size=out_size, mode=mode)
    O, T = int(idx.shape[0]), int(idx.shape[1])
    combined: list[dict[int, np.float32]] = [dict() for _ in range(in_size)]
    for o in range(O):
        for t in range(T):
            i = int(idx[o, t])
            d = combined[i]
            d[o] = np.float32(d.get(o, np.float32(0.0)) + np.float32(w[o, t]))
    maxt = max((len(d) for d in combined), default=0)
    tidx_pad = np.zeros((in_size, maxt), dtype=np.int32)
    tw_pad = np.zeros((in_size, maxt), dtype=np.float32)
    counts = np.zeros((in_size,), dtype=np.int32)
    starts = np.zeros((in_size,), dtype=np.int32)
    flat_idx: list[int] = []
    flat_w: list[np.float32] = []
    pos = 0
    for i in range(in_size):
        outs = sorted(combined[i].keys())  # ascending forward-output index
        counts[i] = len(outs)
        starts[i] = pos
        for k, o in enumerate(outs):
            wv = np.float32(combined[i][o])
            tidx_pad[i, k] = o
            tw_pad[i, k] = wv
            flat_idx.append(int(o))
            flat_w.append(wv)
            pos += 1
    result = {
        "in_size": in_size,
        "out_size": out_size,
        "maxt": int(maxt),
        "tidx_pad": tidx_pad,
        "tw_pad": tw_pad,
        "counts": counts,
        "starts": starts,
        "flat_idx": (
            np.asarray(flat_idx, dtype=np.int32) if flat_idx else np.zeros((0,), np.int32)
        ),
        "flat_w": (
            np.asarray(flat_w, dtype=np.float32) if flat_w else np.zeros((0,), np.float32)
        ),
    }
    _transpose_np_cache[key] = result
    return result


def _transpose_axis_numpy(inp_l_sout_r: np.ndarray, tables: dict[str, Any]) -> np.ndarray:
    """Deterministic sequential (increasing forward-output) transpose over the mid axis.

    ``inp`` is ``(L, S_out, R)`` float32 -> returns ``(L, S_in, R)``. Each ``s_in``
    accumulates its taps in ascending ``s_out`` order with float32 multiply-then-add,
    matching the Metal kernel's per-thread ``contract(off)`` accumulation bit-for-bit.
    Padded taps carry weight 0 (adding ±0.0 preserves the accumulator exactly), so
    the padded vectorized form == the kernel's count-bounded loop.
    """

    inp = np.asarray(inp_l_sout_r, dtype=np.float32)
    L, _Sout, R = int(inp.shape[0]), int(inp.shape[1]), int(inp.shape[2])
    Sin = int(tables["in_size"])
    maxt = int(tables["maxt"])
    tidx = tables["tidx_pad"]
    tw = tables["tw_pad"]
    acc = np.zeros((L, Sin, R), dtype=np.float32)
    for k in range(maxt):
        gathered = inp[:, tidx[:, k], :]  # (L, Sin, R)
        wk = tw[:, k][None, :, None].astype(np.float32)  # (1, Sin, 1)
        acc = (acc + (wk * gathered).astype(np.float32)).astype(np.float32)
    return acc


def _fused_r_vjp_numpy_einsum_legacy(
    x_nhwc: np.ndarray,
    cotangent_nhwc: np.ndarray,
    *,
    camera_hw: tuple[int, int] = CAMERA_HW,
    output_hw: tuple[int, int] = SCORER_HW,
    ste_round: bool = True,
) -> np.ndarray:
    """LEGACY BLAS-``einsum`` VJP (dense matrices, blocked-sgemm accumulation order).

    Preserved ONLY for the transparent residual report + regression guard vs the
    live deterministic ``fused_r_vjp_numpy``. NOT metal-reproducible (accumulation
    order is a BLAS artifact); do NOT use as the parity target.
    """

    x = np.asarray(x_nhwc, dtype=np.float32)
    g = np.asarray(cotangent_nhwc, dtype=np.float32)
    if x.ndim < 4 or x.shape[-1] != 3:
        raise ValueError(f"expected (..., H, W, 3) RGB, got {x.shape}")
    lead = x.shape[:-3]
    Hin, Win = int(x.shape[-3]), int(x.shape[-2])
    cam_h, cam_w = int(camera_hw[0]), int(camera_hw[1])
    out_h, out_w = int(output_hw[0]), int(output_hw[1])
    xf = x.reshape((-1, Hin, Win, 3))
    gf = g.reshape((-1, out_h, out_w, 3))

    M_up_h = _axis_matrix_numpy(in_size=Hin, out_size=cam_h, mode="bicubic")     # (cam_h, Hin)
    M_up_w = _axis_matrix_numpy(in_size=Win, out_size=cam_w, mode="bicubic")     # (cam_w, Win)
    M_dn_h = _axis_matrix_numpy(in_size=cam_h, out_size=out_h, mode="bilinear")  # (out_h, cam_h)
    M_dn_w = _axis_matrix_numpy(in_size=cam_w, out_size=out_w, mode="bilinear")  # (out_w, cam_w)

    cam = np.einsum("oi,niwc->nowc", M_up_h, xf, optimize=True).astype(np.float32)
    cam = np.einsum("oj,nhjc->nhoc", M_up_w, cam, optimize=True).astype(np.float32)
    if ste_round:
        clip_mask = ((cam > 0.0) & (cam < 255.0)).astype(np.float32)
    else:
        clip_mask = np.ones_like(cam, dtype=np.float32)

    g_cam = np.einsum("oj,nhoc->nhjc", M_dn_w, gf, optimize=True).astype(np.float32)   # W^T
    g_cam = np.einsum("oi,nowc->niwc", M_dn_h, g_cam, optimize=True).astype(np.float32)  # H^T
    g_cam = (g_cam * clip_mask).astype(np.float32)
    g_x = np.einsum("oj,nhoc->nhjc", M_up_w, g_cam, optimize=True).astype(np.float32)  # W^T
    g_x = np.einsum("oi,nowc->niwc", M_up_h, g_x, optimize=True).astype(np.float32)    # H^T
    return g_x.reshape((*lead, Hin, Win, 3)).astype(np.float32)


def fused_r_vjp_numpy(
    x_nhwc: np.ndarray,
    cotangent_nhwc: np.ndarray,
    *,
    camera_hw: tuple[int, int] = CAMERA_HW,
    output_hw: tuple[int, int] = SCORER_HW,
    ste_round: bool = True,
) -> np.ndarray:
    """DETERMINISTIC analytic VJP oracle for the contest-faithful R.

    ``grad_x = Up^T( clip_mask * Down^T( cotangent ) )``, where ``clip_mask = 1``
    where the pre-round camera value is strictly inside ``(0, 255)`` and 0 otherwise
    (STE passes the round through; clip is the only nonlinearity with a non-trivial
    Jacobian). ``Down^T`` / ``Up^T`` are the transposes of the two separable resamples,
    computed axis-by-axis in the SAME order as the forward (``Down^T = H_down^T ∘
    W_down^T`` reverses ``Down = W_down ∘ H_down``), with a float32 intermediate
    between axes (mirroring ``_resample_pair_numpy``).

    Accumulation is DETERMINISTIC (per-input, ascending forward-output index, float32
    multiply-then-add — see ``_transpose_sorted_taps`` note) so the Metal transpose
    VJP matches this bit-for-bit. The clip mask is read from the RAW-tap forward-up
    camera (``_resample_pair_numpy(..., do_round=False)``) — identical to
    ``fused_r_forward_numpy``'s pre-round value and to the metal forward kernel — so
    the numpy and metal masks agree even at overshoot pixels near the clip boundary.
    """

    x = np.asarray(x_nhwc, dtype=np.float32)
    g = np.asarray(cotangent_nhwc, dtype=np.float32)
    if x.ndim < 4 or x.shape[-1] != 3:
        raise ValueError(f"expected (..., H, W, 3) RGB, got {x.shape}")
    lead = x.shape[:-3]
    Hin, Win = int(x.shape[-3]), int(x.shape[-2])
    cam_h, cam_w = int(camera_hw[0]), int(camera_hw[1])
    out_h, out_w = int(output_hw[0]), int(output_hw[1])
    xf = x.reshape((-1, Hin, Win, 3))
    n = int(xf.shape[0])
    gf = g.reshape((-1, out_h, out_w, 3))

    dn_w = _transpose_sorted_taps(in_size=cam_w, out_size=out_w, mode="bilinear")
    dn_h = _transpose_sorted_taps(in_size=cam_h, out_size=out_h, mode="bilinear")
    up_w = _transpose_sorted_taps(in_size=Win, out_size=cam_w, mode="bicubic")
    up_h = _transpose_sorted_taps(in_size=Hin, out_size=cam_h, mode="bicubic")

    if ste_round:
        up_h_idx, up_h_w = resize_indices_weights_numpy(in_size=Hin, out_size=cam_h, mode="bicubic")
        up_w_idx, up_w_w = resize_indices_weights_numpy(in_size=Win, out_size=cam_w, mode="bicubic")
        cam = _resample_pair_numpy(
            xf, hidx=up_h_idx, hw=up_h_w, widx=up_w_idx, ww=up_w_w, do_round=False
        )  # (n, cam_h, cam_w, 3) RAW pre-round/pre-clip
        clip_mask = ((cam > 0.0) & (cam < 255.0)).astype(np.float32)
    else:
        clip_mask = None

    # Down^T = H_down^T ∘ W_down^T (reverse of Down = W_down ∘ H_down).
    a = _transpose_axis_numpy(gf.reshape(n * out_h, out_w, 3), dn_w).reshape(n, out_h, cam_w, 3)
    b = _transpose_axis_numpy(a.reshape(n, out_h, cam_w * 3), dn_h).reshape(n, cam_h, cam_w, 3)
    if clip_mask is not None:
        b = (b * clip_mask).astype(np.float32)
    # Up^T = H_up^T ∘ W_up^T.
    d = _transpose_axis_numpy(b.reshape(n * cam_h, cam_w, 3), up_w).reshape(n, cam_h, Win, 3)
    g_x = _transpose_axis_numpy(d.reshape(n, cam_h, Win * 3), up_h).reshape(n, Hin, Win, 3)
    return g_x.reshape((*lead, Hin, Win, 3)).astype(np.float32)


# --------------------------------------------------------------------------- #
# METAL FORWARD KERNEL — fused separable resample (+ optional clip/round).
# --------------------------------------------------------------------------- #

# One thread per OUTPUT element (n, ho, wo, c). Separable: W-outer, H-inner tap
# loop (== MLX H-pass-then-W-pass order => bit-faithful). do_round -> clip[0,255]
# then rint (round-half-to-even, matching mx.round).
#
# ``#pragma clang fp contract(off)`` is LOAD-BEARING for bit-identity: Metal is
# clang-based and defaults to FP contraction (fuses ``w*x + acc`` into a single
# ``fma`` with no intermediate rounding). numpy computes the separable resample as
# separate multiply-then-add (each product rounded to float32 before the add), so
# an FMA kernel diverges by sub-ULP on the float down-sample AND flips a handful of
# camera pixels across the round-half boundary (measured on M5 Max: 3.05e-05 on 14%
# of down-sampled pixels + a few 1-LSB round flips). Disabling contraction makes the
# kernel bit-IDENTICAL (max|Δ|=0, 100% exact) to the numpy-fp32 authority, matching
# the MLX production R. ``#pragma METAL fp math_mode(safe)`` does NOT disable
# contraction; only ``clang fp contract(off)`` does (verified on this host).
_RESAMPLE_SRC = """
    #pragma clang fp contract(off)
    uint gid = thread_position_in_grid.x;
    int N     = dims[0];
    int Hin   = dims[1];
    int Win   = dims[2];
    int C     = dims[3];
    int Hout  = dims[4];
    int Wout  = dims[5];
    int tapsH = dims[6];
    int tapsW = dims[7];
    int do_round = dims[8];
    int total = N * Hout * Wout * C;
    if (gid >= (uint)total) return;

    int c  = gid % C;
    int wo = (gid / C) % Wout;
    int ho = (gid / (C * Wout)) % Hout;
    int n  =  gid / (C * Wout * Hout);

    float acc = 0.0f;
    for (int tw = 0; tw < tapsW; ++tw) {
        int wi   = widx[wo * tapsW + tw];
        float wwt = ww[wo * tapsW + tw];
        float col = 0.0f;
        for (int th = 0; th < tapsH; ++th) {
            int hi   = hidx[ho * tapsH + th];
            float wht = hw[ho * tapsH + th];
            int xidx = ((n * Hin + hi) * Win + wi) * C + c;
            col += wht * x[xidx];
        }
        acc += wwt * col;
    }
    if (do_round != 0) {
        acc = fmin(fmax(acc, 0.0f), 255.0f);
        acc = rint(acc);
    }
    y[gid] = acc;
"""

_resample_kernel = None
# Cache of per-(in,out,mode) flattened index/weight mx arrays, keyed string.
_table_cache: dict[str, Any] = {}


def _kernel():
    global _resample_kernel
    if _resample_kernel is None:
        import mlx.core as mx

        _resample_kernel = mx.fast.metal_kernel(
            name="fused_r_separable_resample",
            input_names=["x", "hidx", "hw", "widx", "ww", "dims"],
            output_names=["y"],
            source=_RESAMPLE_SRC,
        )
    return _resample_kernel


def metal_fused_r_available() -> bool:
    """True when an MLX GPU (Metal) device is the default device."""

    try:
        import mlx.core as mx

        return mx.default_device().type == mx.gpu
    except Exception:  # pragma: no cover - device introspection guard
        return False


def _axis_tables_mx(*, in_size: int, out_size: int, mode: str):
    """Return flattened (idx int32, w float32) mx arrays for one axis (cached)."""

    import mlx.core as mx

    key = f"{in_size}:{out_size}:{mode}"
    cached = _table_cache.get(key)
    if cached is None:
        idx_np, w_np = resize_indices_weights_numpy(in_size=in_size, out_size=out_size, mode=mode)
        idx = mx.array(np.ascontiguousarray(idx_np.reshape(-1)).astype(np.int32))
        w = mx.array(np.ascontiguousarray(w_np.reshape(-1)).astype(np.float32))
        taps = int(idx_np.shape[1])
        cached = (idx, w, taps)
        _table_cache[key] = cached
    return cached


def _resample_mx(x_bhwc, *, out_h: int, out_w: int, mode: str, do_round: bool):
    """One fused metal resample launch on (B,Hin,Win,C) -> (B,out_h,out_w,C)."""

    import mlx.core as mx

    B, Hin, Win, C = (int(d) for d in x_bhwc.shape)
    h_idx, h_w, taps_h = _axis_tables_mx(in_size=Hin, out_size=out_h, mode=mode)
    w_idx, w_w, taps_w = _axis_tables_mx(in_size=Win, out_size=out_w, mode=mode)
    dims = mx.array(
        np.array(
            [B, Hin, Win, C, out_h, out_w, taps_h, taps_w, 1 if do_round else 0],
            dtype=np.int32,
        )
    )
    total = B * out_h * out_w * C
    (y,) = _kernel()(
        inputs=[x_bhwc, h_idx, h_w, w_idx, w_w, dims],
        output_shapes=[(B, out_h, out_w, C)],
        output_dtypes=[x_bhwc.dtype],
        grid=(int(total), 1, 1),
        threadgroup=(256, 1, 1),
    )
    return y


def _fused_r_metal_forward(x_nhwc, *, camera_hw, output_hw, ste_round: bool):
    """Fused metal forward (two launches): up+clip+round @ camera, then down."""

    import mlx.core as mx

    if x_nhwc.ndim < 4 or int(x_nhwc.shape[-1]) != 3:
        raise ValueError(f"expected (..., H, W, 3) RGB, got {tuple(x_nhwc.shape)}")
    lead = tuple(int(d) for d in x_nhwc.shape[:-3])
    Hin, Win = int(x_nhwc.shape[-3]), int(x_nhwc.shape[-2])
    flat = mx.reshape(x_nhwc, (-1, Hin, Win, 3))
    cam_h, cam_w = int(camera_hw[0]), int(camera_hw[1])
    out_h, out_w = int(output_hw[0]), int(output_hw[1])
    camera = _resample_mx(flat, out_h=cam_h, out_w=cam_w, mode="bicubic", do_round=bool(ste_round))
    if not ste_round:
        camera = mx.clip(camera, 0.0, 255.0)
    out = _resample_mx(camera, out_h=out_h, out_w=out_w, mode="bilinear", do_round=False)
    return mx.reshape(out, (*lead, out_h, out_w, 3))


# --------------------------------------------------------------------------- #
# METAL TRANSPOSE VJP KERNEL — fully-fused deterministic adjoint of the forward.
#
# One thread per INPUT (transpose-output) element ``(l, s_in, r)``. Each input pixel
# gathers its taps ``Σ w * inp[l, s_out, r]`` in ASCENDING forward-output (``s_out``)
# order (== the deterministic numpy ``_transpose_axis_numpy`` accumulation, so the
# kernel is BIT-IDENTICAL to ``fused_r_vjp_numpy`` at atol=0). Each input's sum is
# INDEPENDENT of every other -> **NO ATOMICS** -> DETERMINISTIC (beats the mx.vjp
# scatter ~1-ULP non-determinism floor). ``#pragma clang fp contract(off)`` is
# LOAD-BEARING (same reason as the forward kernel): it forces separate float32
# multiply-then-add so the accumulation matches numpy's ``(acc + w*g)`` term-by-term
# instead of an FMA. The four separable axis-transposes (W_down^T, H_down^T,
# W_up^T, H_up^T) are four launches of THIS one generic kernel over a
# ``(L, S_out, R)`` view, with a float32 intermediate materialized between axes
# (mirroring the numpy oracle's per-axis ``.astype(float32)``); the clip mask
# multiply sits between the Down^T and Up^T pairs. Boundary handling (edges are where
# d_seg lives): taps that clip to the same ``(s_out, s_in)`` pair are PRE-COMBINED at
# table-build time (``_transpose_sorted_taps``) so each pair appears exactly once —
# this is the exact adjoint weight ``∂y[s_out]/∂x[s_in]`` and avoids double-counting
# a clipped edge tap.
# --------------------------------------------------------------------------- #

_TRANSPOSE_SRC = """
    #pragma clang fp contract(off)
    uint gid = thread_position_in_grid.x;
    int L    = dims[0];
    int Sout = dims[1];
    int Sin  = dims[2];
    int R    = dims[3];
    int total = L * Sin * R;
    if (gid >= (uint)total) return;

    int r  = gid % R;
    int si = (gid / R) % Sin;
    int l  = gid / (R * Sin);

    int start = t_start[si];
    int cnt   = t_cnt[si];
    float acc = 0.0f;
    for (int k = 0; k < cnt; ++k) {
        int so  = t_idx[start + k];
        float w = t_w[start + k];
        int xidx = (l * Sout + so) * R + r;
        acc += w * inp[xidx];
    }
    y[gid] = acc;
"""

_transpose_kernel_obj = None
# Cache of per-(in,out,mode) flattened CSR transpose tables as mx arrays.
_mx_transpose_cache: dict[str, Any] = {}


def _transpose_kernel():
    global _transpose_kernel_obj
    if _transpose_kernel_obj is None:
        import mlx.core as mx

        _transpose_kernel_obj = mx.fast.metal_kernel(
            name="fused_r_transpose_resample",
            input_names=["inp", "t_start", "t_cnt", "t_idx", "t_w", "dims"],
            output_names=["y"],
            source=_TRANSPOSE_SRC,
        )
    return _transpose_kernel_obj


def _mx_transpose_tables(*, in_size: int, out_size: int, mode: str):
    """Return (t_start, t_cnt, t_idx, t_w, in_size) CSR transpose tables (cached mx)."""

    import mlx.core as mx

    key = f"{in_size}:{out_size}:{mode}"
    cached = _mx_transpose_cache.get(key)
    if cached is None:
        t = _transpose_sorted_taps(in_size=in_size, out_size=out_size, mode=mode)
        t_start = mx.array(np.ascontiguousarray(t["starts"]).astype(np.int32))
        t_cnt = mx.array(np.ascontiguousarray(t["counts"]).astype(np.int32))
        t_idx = mx.array(np.ascontiguousarray(t["flat_idx"]).astype(np.int32))
        t_w = mx.array(np.ascontiguousarray(t["flat_w"]).astype(np.float32))
        cached = (t_start, t_cnt, t_idx, t_w, int(t["in_size"]))
        _mx_transpose_cache[key] = cached
    return cached


def _transpose_axis_mx(inp_l_sout_r, *, in_size: int, out_size: int, mode: str):
    """One fused metal transpose-resample launch: (L, S_out, R) -> (L, S_in, R).

    Reduces the middle (``S_out`` == forward-output) axis to ``S_in`` (== forward-input
    == transpose-output) via the CSR transpose tables, ascending-``s_out`` order.
    """

    import mlx.core as mx

    L, Sout, R = (int(d) for d in inp_l_sout_r.shape)
    t_start, t_cnt, t_idx, t_w, Sin = _mx_transpose_tables(
        in_size=in_size, out_size=out_size, mode=mode
    )
    dims = mx.array(np.array([L, Sout, Sin, R], dtype=np.int32))
    total = L * Sin * R
    (y,) = _transpose_kernel()(
        inputs=[inp_l_sout_r, t_start, t_cnt, t_idx, t_w, dims],
        output_shapes=[(L, Sin, R)],
        output_dtypes=[inp_l_sout_r.dtype],
        grid=(int(total), 1, 1),
        threadgroup=(256, 1, 1),
    )
    return y


def _fused_r_metal_vjp(x_nhwc, cot_nhwc, *, camera_hw, output_hw, ste_round: bool):
    """Fully-fused metal transpose VJP: grad_x = Up^T( clip_mask * Down^T( cot ) ).

    Bit-identical to the deterministic ``fused_r_vjp_numpy`` (same tables, same
    reshapes, same ascending-``s_out`` accumulation), deterministic (no atomics), and
    the fast backward for the fused-R custom function.
    """

    import mlx.core as mx

    if x_nhwc.ndim < 4 or int(x_nhwc.shape[-1]) != 3:
        raise ValueError(f"expected (..., H, W, 3) RGB, got {tuple(x_nhwc.shape)}")
    lead = tuple(int(d) for d in x_nhwc.shape[:-3])
    Hin, Win = int(x_nhwc.shape[-3]), int(x_nhwc.shape[-2])
    cam_h, cam_w = int(camera_hw[0]), int(camera_hw[1])
    out_h, out_w = int(output_hw[0]), int(output_hw[1])
    xf = mx.reshape(x_nhwc, (-1, Hin, Win, 3))
    n = int(xf.shape[0])
    gf = mx.reshape(cot_nhwc, (-1, out_h, out_w, 3))

    # Down^T = H_down^T ∘ W_down^T (reverse of Down = W_down ∘ H_down).
    a = _transpose_axis_mx(
        mx.reshape(gf, (n * out_h, out_w, 3)), in_size=cam_w, out_size=out_w, mode="bilinear"
    )
    a = mx.reshape(a, (n, out_h, cam_w, 3))
    b = _transpose_axis_mx(
        mx.reshape(a, (n, out_h, cam_w * 3)), in_size=cam_h, out_size=out_h, mode="bilinear"
    )
    b = mx.reshape(b, (n, cam_h, cam_w, 3))

    if ste_round:
        # RAW pre-round camera (bit-identical to the numpy oracle's mask source and
        # to fused_r_forward_numpy's pre-round value).
        cam = _resample_mx(xf, out_h=cam_h, out_w=cam_w, mode="bicubic", do_round=False)
        mask = mx.logical_and(cam > 0.0, cam < 255.0).astype(mx.float32)
        b = b * mask

    # Up^T = H_up^T ∘ W_up^T.
    d = _transpose_axis_mx(
        mx.reshape(b, (n * cam_h, cam_w, 3)), in_size=Win, out_size=cam_w, mode="bicubic"
    )
    d = mx.reshape(d, (n, cam_h, Win, 3))
    g_x = _transpose_axis_mx(
        mx.reshape(d, (n, cam_h, Win * 3)), in_size=Hin, out_size=cam_h, mode="bicubic"
    )
    g_x = mx.reshape(g_x, (n, Hin, Win, 3))
    return mx.reshape(g_x, (*lead, Hin, Win, 3))


# --------------------------------------------------------------------------- #
# CUSTOM FUNCTION — metal forward + metal transpose-VJP backward.
# --------------------------------------------------------------------------- #


def make_fused_r_roundtrip(
    *,
    camera_hw: tuple[int, int] = CAMERA_HW,
    output_hw: tuple[int, int] = SCORER_HW,
    ste_round: bool = True,
):
    """Return a config-bound ``@mx.custom_function`` ``fn(x)->R(x)``.

    Why a factory (mirrors ``metal_grouped_conv_backward.make_grouped_conv2d_nhwc``):
    MLX's ``custom_function.vjp`` only receives ``(primals, cotangent, output)`` and
    does NOT forward keyword config, so the camera/output/ste config is bound by
    closure. FORWARD = the fused metal kernels (fast, requires a Metal default
    device). BACKWARD = ``mx.vjp`` of the bit-faithful pure-MLX oracle
    ``apply_contest_faithful_roundtrip_nhwc`` (correct STE/clip/transpose grad;
    near-zero cross-chip risk).
    """

    import mlx.core as mx

    cam = (int(camera_hw[0]), int(camera_hw[1]))
    out_hw = (int(output_hw[0]), int(output_hw[1]))
    ste = bool(ste_round)

    @mx.custom_function
    def fn(x):
        return _fused_r_metal_forward(x, camera_hw=cam, output_hw=out_hw, ste_round=ste)

    @fn.vjp
    def _fn_vjp(primals, cotangent, output):
        # MLX 0.31.2 calling convention: for a SINGLE-argument custom_function,
        # ``primals`` is passed as the RAW array (NOT a length-1 tuple ``(x,)``);
        # for multi-arg it is a tuple. The prior ``(x,) = primals`` unpacked the
        # array's leading dim -> ``ValueError: too many values to unpack`` (crash
        # whenever B != 1, silent wrong-shape when B == 1). Handle both conventions.
        x = primals[0] if isinstance(primals, (tuple, list)) else primals
        # cotangent mirrors the primal packing (raw for single output here).
        cot = cotangent[0] if isinstance(cotangent, (tuple, list)) else cotangent
        # BACKWARD = the fully-fused metal transpose VJP. Deterministic (no atomics)
        # and BIT-IDENTICAL to the deterministic numpy authority ``fused_r_vjp_numpy``
        # (P2b). This replaces the prior ``mx.vjp`` of the pure-MLX oracle, which was
        # correct but re-ran the slow MLX forward + carried a ~1-ULP scatter
        # non-determinism floor.
        gx = _fused_r_metal_vjp(x, cot, camera_hw=cam, output_hw=out_hw, ste_round=ste)
        return (gx,)

    return fn


def fused_r_roundtrip(
    x_nhwc,
    *,
    camera_hw: tuple[int, int] = CAMERA_HW,
    output_hw: tuple[int, int] = SCORER_HW,
    ste_round: bool = True,
):
    """Convenience: build the config-bound custom fn and apply it (metal forward).

    Drop-in for ``apply_contest_faithful_roundtrip_nhwc`` when the
    ``--fused-r-kernel`` flag is ON and a Metal device is active.
    """

    fn = make_fused_r_roundtrip(camera_hw=camera_hw, output_hw=output_hw, ste_round=ste_round)
    return fn(x_nhwc)


def fused_r_roundtrip_reference(
    x_nhwc,
    *,
    camera_hw: tuple[int, int] = CAMERA_HW,
    output_hw: tuple[int, int] = SCORER_HW,
    ste_round: bool = True,
):
    """Pure-MLX reference path (flag OFF / non-Metal device fallback)."""

    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        apply_contest_faithful_roundtrip_nhwc,
    )

    return apply_contest_faithful_roundtrip_nhwc(
        x_nhwc,
        camera_hw=(int(camera_hw[0]), int(camera_hw[1])),
        output_hw=(int(output_hw[0]), int(output_hw[1])),
        ste_round=bool(ste_round),
    )


# --------------------------------------------------------------------------- #
# PER-CHIP PARITY GUARD (issue #2205: metal_kernel can be wrong on some chips).
# --------------------------------------------------------------------------- #


def assert_metal_matches_cpu_oracle(
    *,
    seed: int = 0,
    in_hw: tuple[int, int] = (48, 64),
    camera_hw: tuple[int, int] = (110, 146),
    output_hw: tuple[int, int] = (48, 64),
    batch: int = 2,
    fwd_atol: float = 0.0,
    grad_atol: float = 0.0,
) -> dict[str, Any]:
    """GPU-gated per-chip correctness check: metal forward AND VJP == numpy oracle.

    Runs the fused-R metal FORWARD and the fused metal transpose VJP on the current
    (Metal) device and compares BOTH to the deterministic numpy oracle BIT-FOR-BIT
    (``fwd_atol=grad_atol=0`` default — same float ops, same ascending-output order).
    The VJP is checked both directly (``_fused_r_metal_vjp``) and through the custom
    function (``mx.vjp(fn, ...)``). Small shapes by default (<2 GB, <1 s). RAISES on
    mismatch so a miswired/cross-chip-broken kernel cannot be promoted. Call from the
    gated GPU harness only (requires a Metal default device).
    """

    import mlx.core as mx

    if not metal_fused_r_available():
        raise RuntimeError(
            "assert_metal_matches_cpu_oracle requires a Metal (GPU) default device; "
            "current default is "
            f"{mx.default_device()}"
        )
    rng = np.random.default_rng(int(seed))
    x_np = (rng.random((batch, in_hw[0], in_hw[1], 3)) * 255.0).astype(np.float32)
    # numpy oracle forward
    y_oracle = fused_r_forward_numpy(
        x_np, camera_hw=camera_hw, output_hw=output_hw, ste_round=True
    )
    # metal forward
    x_mx = mx.array(x_np)
    y_metal = np.asarray(
        _fused_r_metal_forward(x_mx, camera_hw=camera_hw, output_hw=output_hw, ste_round=True)
    )
    fwd_max = float(np.max(np.abs(y_metal - y_oracle))) if y_metal.size else 0.0

    # VJP: metal transpose kernel vs the DETERMINISTIC numpy analytic transpose (atol=0).
    cot_np = rng.standard_normal(y_oracle.shape).astype(np.float32)
    cot_mx = mx.array(cot_np)
    gx_direct = np.asarray(
        _fused_r_metal_vjp(x_mx, cot_mx, camera_hw=camera_hw, output_hw=output_hw, ste_round=True)
    )
    fn = make_fused_r_roundtrip(camera_hw=camera_hw, output_hw=output_hw, ste_round=True)
    _, (gx_fn_mx,) = mx.vjp(fn, (x_mx,), (cot_mx,))
    gx_via_fn = np.asarray(gx_fn_mx)
    gx_oracle = fused_r_vjp_numpy(
        x_np, cot_np, camera_hw=camera_hw, output_hw=output_hw, ste_round=True
    )
    grad_max = float(np.max(np.abs(gx_direct - gx_oracle))) if gx_direct.size else 0.0
    grad_via_fn_max = (
        float(np.max(np.abs(gx_via_fn - gx_oracle))) if gx_via_fn.size else 0.0
    )

    fwd_ok = bool(fwd_max <= float(fwd_atol))
    grad_ok = bool(grad_max <= float(grad_atol) and grad_via_fn_max <= float(grad_atol))
    result = {
        "device": str(mx.default_device()),
        "forward_max_abs_delta": fwd_max,
        "forward_bit_identical": fwd_ok,
        "grad_max_abs_delta": grad_max,
        "grad_via_custom_fn_max_abs_delta": grad_via_fn_max,
        "grad_bit_identical": grad_ok,
        # legacy key preserved for existing callers/tests (now means bit-identity @ atol=0)
        "grad_within_tol": grad_ok,
        "fwd_atol": float(fwd_atol),
        "grad_atol": float(grad_atol),
    }
    if not fwd_ok:
        raise AssertionError(
            f"fused-R metal FORWARD diverges from numpy oracle on {result['device']}: "
            f"max|Δ|={fwd_max} > {fwd_atol} (cross-chip correctness guard, issue #2205)"
        )
    if not grad_ok:
        raise AssertionError(
            f"fused-R metal VJP diverges from the deterministic numpy analytic VJP on "
            f"{result['device']}: direct max|Δ|={grad_max}, via-custom-fn max|Δ|="
            f"{grad_via_fn_max} > {grad_atol} (cross-chip correctness guard, issue #2205)"
        )
    return result


def assert_metal_vjp_deterministic(
    *,
    seed: int = 0,
    in_hw: tuple[int, int] = (48, 64),
    camera_hw: tuple[int, int] = (110, 146),
    output_hw: tuple[int, int] = (48, 64),
    batch: int = 2,
    repeats: int = 3,
) -> dict[str, Any]:
    """GPU-gated determinism check: the metal transpose VJP is bit-identical across runs.

    Runs ``_fused_r_metal_vjp`` ``repeats`` times on the same input and asserts every
    run is BIT-IDENTICAL (max|Δ|=0). This is the atomics-free-scatter guarantee (P2b):
    the fused metal transpose VJP beats the ~1-ULP non-determinism floor of the prior
    ``mx.vjp`` scatter backward. RAISES on any run-to-run difference.
    """

    import mlx.core as mx

    if not metal_fused_r_available():
        raise RuntimeError(
            "assert_metal_vjp_deterministic requires a Metal (GPU) default device; "
            f"current default is {mx.default_device()}"
        )
    rng = np.random.default_rng(int(seed))
    x_np = (rng.random((batch, in_hw[0], in_hw[1], 3)) * 255.0).astype(np.float32)
    y_shape = fused_r_forward_numpy(x_np, camera_hw=camera_hw, output_hw=output_hw).shape
    cot_np = rng.standard_normal(y_shape).astype(np.float32)
    x_mx = mx.array(x_np)
    cot_mx = mx.array(cot_np)
    runs = []
    for _ in range(int(repeats)):
        runs.append(
            np.asarray(
                _fused_r_metal_vjp(
                    x_mx, cot_mx, camera_hw=camera_hw, output_hw=output_hw, ste_round=True
                )
            )
        )
    max_pairdiff = 0.0
    for i in range(1, len(runs)):
        max_pairdiff = max(max_pairdiff, float(np.max(np.abs(runs[i] - runs[0]))))
    deterministic = bool(max_pairdiff == 0.0)
    result = {
        "device": str(mx.default_device()),
        "repeats": int(repeats),
        "max_run_to_run_abs_delta": max_pairdiff,
        "deterministic_bit_identical": deterministic,
    }
    if not deterministic:
        raise AssertionError(
            f"fused-R metal VJP is NON-deterministic on {result['device']}: "
            f"run-to-run max|Δ|={max_pairdiff} != 0 (expected atomics-free bit-identity)"
        )
    return result
