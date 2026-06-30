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

Design (forward = fast metal; backward = correct MLX autodiff of the oracle)
---------------------------------------------------------------------------
* FORWARD: two ``mx.fast.metal_kernel`` launches (up+clip+round, then down),
  W-outer / H-inner tap order == MLX's H-pass-then-W-pass order => bit-faithful.
* VJP: computed via ``mx.vjp`` of the bit-faithful pure-MLX oracle
  ``apply_contest_faithful_roundtrip_nhwc`` (STE passthrough for the uint8 round +
  clip subgradient + transpose of the two linear resamples). Rationale: unlike the
  strided grouped-conv backward (which is *numerically wrong* natively and so
  NEEDS a custom Metal grad kernel), the resize backward is *correct* in native
  MLX (only the forward is slow), and the R backward is a small fraction of the
  step vs the grouped-conv backward (the >97% lever, handled separately). So the
  honest, low-cross-chip-risk choice is custom-metal FORWARD + autodiff-faithful
  backward. A fully-fused metal transpose VJP is a documented future extension
  (memo P2b). The numpy ``fused_r_vjp_numpy`` is the analytic-transpose oracle
  used to validate gradient correctness independent of MLX.

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
    """Dense (out_size, in_size) float32 resample matrix (sum of clipped taps)."""

    idx, w = resize_indices_weights_numpy(in_size=in_size, out_size=out_size, mode=mode)
    out_size = int(out_size)
    in_size = int(in_size)
    mat = np.zeros((out_size, in_size), dtype=np.float32)
    for o in range(out_size):
        for t in range(idx.shape[1]):
            mat[o, int(idx[o, t])] += np.float32(w[o, t])
    return mat


def fused_r_vjp_numpy(
    x_nhwc: np.ndarray,
    cotangent_nhwc: np.ndarray,
    *,
    camera_hw: tuple[int, int] = CAMERA_HW,
    output_hw: tuple[int, int] = SCORER_HW,
    ste_round: bool = True,
) -> np.ndarray:
    """Analytic VJP oracle for the contest-faithful R.

    grad_x = Up^T( clip_mask * Down^T( cotangent ) ), where clip_mask = 1 where the
    pre-round camera value is strictly inside (0, 255) and 0 otherwise (STE passes
    the round through; clip is the only nonlinearity with a non-trivial Jacobian).
    Down^T / Up^T are the transposes of the separable bilinear/bicubic resamples.
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

    # Forward camera pre-round value (for the clip mask).
    cam = np.einsum("oi,niwc->nowc", M_up_h, xf, optimize=True).astype(np.float32)
    cam = np.einsum("oj,nhjc->nhoc", M_up_w, cam, optimize=True).astype(np.float32)
    if ste_round:
        clip_mask = ((cam > 0.0) & (cam < 255.0)).astype(np.float32)
    else:
        clip_mask = np.ones_like(cam, dtype=np.float32)

    # Down^T: transpose of (H-down then W-down) == (W-down^T then H-down^T).
    g_cam = np.einsum("oj,nhoc->nhjc", M_dn_w, gf, optimize=True).astype(np.float32)   # W^T
    g_cam = np.einsum("oi,nowc->niwc", M_dn_h, g_cam, optimize=True).astype(np.float32)  # H^T
    g_cam = (g_cam * clip_mask).astype(np.float32)
    # Up^T: transpose of (H-up then W-up) == (W-up^T then H-up^T).
    g_x = np.einsum("oj,nhoc->nhjc", M_up_w, g_cam, optimize=True).astype(np.float32)  # W^T
    g_x = np.einsum("oi,nowc->niwc", M_up_h, g_x, optimize=True).astype(np.float32)    # H^T
    return g_x.reshape((*lead, Hin, Win, 3)).astype(np.float32)


# --------------------------------------------------------------------------- #
# METAL FORWARD KERNEL — fused separable resample (+ optional clip/round).
# --------------------------------------------------------------------------- #

# One thread per OUTPUT element (n, ho, wo, c). Separable: W-outer, H-inner tap
# loop (== MLX H-pass-then-W-pass order => bit-faithful). do_round -> clip[0,255]
# then rint (round-half-to-even, matching mx.round).
_RESAMPLE_SRC = """
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
# CUSTOM FUNCTION — metal forward + autodiff-faithful (oracle) backward.
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

    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        apply_contest_faithful_roundtrip_nhwc,
    )

    cam = (int(camera_hw[0]), int(camera_hw[1]))
    out_hw = (int(output_hw[0]), int(output_hw[1]))
    ste = bool(ste_round)

    def _reference(z):
        return apply_contest_faithful_roundtrip_nhwc(
            z, camera_hw=cam, output_hw=out_hw, ste_round=ste
        )

    @mx.custom_function
    def fn(x):
        return _fused_r_metal_forward(x, camera_hw=cam, output_hw=out_hw, ste_round=ste)

    @fn.vjp
    def _fn_vjp(primals, cotangent, output):
        (x,) = primals
        _, (gx,) = mx.vjp(_reference, (x,), (cotangent,))
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
    grad_rtol: float = 2e-3,
    grad_atol: float = 2e-4,
) -> dict[str, Any]:
    """GPU-gated per-chip correctness check: metal forward == numpy oracle.

    Runs the fused-R metal FORWARD on the current (Metal) device and compares to
    the numpy oracle bit-for-bit (``fwd_atol=0`` default — same float ops, same
    order). Also checks the custom-function VJP against the numpy analytic VJP
    within tol. Small shapes by default (<2 GB, <1 s). RAISES on mismatch so a
    miswired/cross-chip-broken kernel cannot be promoted. Call from the gated GPU
    harness only (requires a Metal default device).
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

    # VJP: custom fn vs numpy analytic transpose.
    fn = make_fused_r_roundtrip(camera_hw=camera_hw, output_hw=output_hw, ste_round=True)
    cot_np = rng.standard_normal(y_oracle.shape).astype(np.float32)
    cot_mx = mx.array(cot_np)
    _, (gx_mx,) = mx.vjp(fn, (x_mx,), (cot_mx,))
    gx_metal = np.asarray(gx_mx)
    gx_oracle = fused_r_vjp_numpy(
        x_np, cot_np, camera_hw=camera_hw, output_hw=output_hw, ste_round=True
    )
    grad_max = float(np.max(np.abs(gx_metal - gx_oracle))) if gx_metal.size else 0.0
    grad_ok = bool(
        np.allclose(gx_metal, gx_oracle, rtol=float(grad_rtol), atol=float(grad_atol))
    )

    fwd_ok = bool(fwd_max <= float(fwd_atol))
    result = {
        "device": str(mx.default_device()),
        "forward_max_abs_delta": fwd_max,
        "forward_bit_identical": fwd_ok,
        "grad_max_abs_delta": grad_max,
        "grad_within_tol": grad_ok,
        "fwd_atol": float(fwd_atol),
        "grad_rtol": float(grad_rtol),
        "grad_atol": float(grad_atol),
    }
    if not fwd_ok:
        raise AssertionError(
            f"fused-R metal FORWARD diverges from numpy oracle on {result['device']}: "
            f"max|Δ|={fwd_max} > {fwd_atol} (cross-chip correctness guard, issue #2205)"
        )
    if not grad_ok:
        raise AssertionError(
            f"fused-R VJP diverges from numpy analytic VJP on {result['device']}: "
            f"max|Δ|={grad_max} (rtol={grad_rtol}, atol={grad_atol})"
        )
    return result
