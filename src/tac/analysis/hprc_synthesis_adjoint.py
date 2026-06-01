# SPDX-License-Identifier: MIT
"""G3 — exact HPRC synthesis adjoint A^T (Daubechies representation-domain gate).

The score-exact saliency producer (``tac.analysis.score_exact_saliency``) emits
``s_seg`` (P18 flip-risk) and ``s_pose`` (P19 Fisher) in **camera-native PIXEL
space**. But the HPRC carrier allocates bits in its **representation domain** —
the residual-token grid ``q[frame, gh, gw, c]`` and the per-frame latent vector
``latent[frame, k]``. The reverse-waterfill (``rate_collapse`` /
``joint_p18_p19_waterfill``) protects high-importance TOKENS and dead-zones the
rest, so the pixel-space saliency must be pushed into the token/coefficient
domain *exactly*, via the adjoint of the decode synthesis ``A`` (the
``latent/token -> frame`` linear map). If the adjoint is approximate or the
basis is not orthonormal, the saliency is mis-attributed and the allocator
spends bytes on the WRONG coefficients — silently. (Daubechies' GAP-3 gate, per
``.omx/research/council_t3_score_exact_rd_oracle_keystone_ratification_20260601.md``
binding Revision 2(c).)

What ``A`` actually is for HPRC's ``render_compact_receiver_frame_batch``
(source-inspected ``learned_receiver.py``):

    frame[f,H,W,c] = mean[H,W,c]
        + latent_gain * (latent[f,:] @ basis[:,H,W,c])             # latent stage
        + residual_gain * selector[f] * output_resize(            # residual stage
              scale * nearest_resize(residual_q[f,gh,gw,c]) )

The ``mean`` is a constant offset (zero Jacobian). Both ``nearest_resize`` and
``bilinear_resize`` are pure **linear gather** operators (index/weight maps, no
nonlinearity). The final ``clip``+``round`` is a non-linear OUTPUT clamp; it is
NOT part of the bit-allocation Jacobian (the allocator linearizes the score
about the operating point, and the council's keystone is verified in pixel space
on the un-clamped synthesis). The adjoint of a gather is a **scatter-add**: the
transpose accumulates each output cell's saliency back into the source cell that
fed it. For a nearest-resize on a grid this is exactly block sum-pooling — the
orthonormal-grid synthesis adjoint Daubechies named.

This module provides:

  * ``nearest_resize_adjoint`` / ``bilinear_resize_adjoint`` — exact transposes of
    the two HPRC resize gathers, proven with the canonical dot-product test
    ``<A x, y> == <x, A^T y>``.
  * ``push_pixel_saliency_to_residual_grid`` — A_resid^T applied to a pixel-space
    saliency surface, yielding ``(frames, grid_h, grid_w[, channels])``
    coefficient-domain saliency the ``rate_collapse`` importance consumer
    accepts directly.
  * ``push_pixel_saliency_to_latent`` — A_latent^T (= basis @ pixel_saliency)
    yielding per-latent-dim saliency.
  * ``adjoint_dotproduct_residual`` — the NO-FAKE exactness guard: builds A x in
    the pixel domain, A^T y in the token domain, and returns
    ``|<Ax,y> - <x,A^Ty>| / scale``. A non-adjoint transform fails this.

All outputs are ``[macOS-CPU advisory]`` / NON-PROMOTABLE — this is a
COMPRESS-SIDE allocation producer; nothing crosses the receiver boundary except
``archive.zip`` + the scorer-free ``inflate`` runtime (per
``build_saliency_verification_contract`` ``contest_compliance``).

References:
  - Daubechies 1988 "Orthonormal bases of compactly supported wavelets" — the
    synthesis/analysis adjoint pair for an orthonormal basis (here the trivial
    block-replication / block-sum-pool grid basis).
  - The canonical adjoint dot-product test ``<A x, y>_Y == <x, A^T y>_X`` (the
    definition of the adjoint operator; a transform that fails it is not A^T).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Exact transposes of the two HPRC resize GATHER operators.
#
# Forward (gather) for nearest-resize, source (sh, sw) -> dest (dh, dw):
#     y_idx[i] = (i * sh) // dh ; x_idx[j] = (j * sw) // dw
#     dest[i, j, c] = src[y_idx[i], x_idx[j], c]
# This is a linear gather: dest = G @ src (G is a {0,1} selection matrix per
# axis). Its adjoint G^T is the scatter-add: A^T y accumulates every dest cell's
# value into the (single) source cell it gathered from. We mirror the EXACT
# index arithmetic of ``learned_receiver._nearest_resize`` so the adjoint is the
# transpose of the SHIPPED forward op, not an idealized one.
# ---------------------------------------------------------------------------


def _nearest_resize_index_maps(
    src_h: int, src_w: int, dst_h: int, dst_w: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exact index maps used by ``learned_receiver._nearest_resize``."""
    y_idx = (np.arange(dst_h, dtype=np.int64) * src_h // dst_h).clip(0, src_h - 1)
    x_idx = (np.arange(dst_w, dtype=np.int64) * src_w // dst_w).clip(0, src_w - 1)
    return y_idx, x_idx


def nearest_resize_forward(src: np.ndarray, dst_h: int, dst_w: int) -> np.ndarray:
    """Forward nearest-resize gather (mirrors learned_receiver._nearest_resize).

    ``src`` is ``(sh, sw, c)`` or ``(F, sh, sw, c)``. The leading frame dim is
    passed through unchanged.
    """
    arr = np.asarray(src, dtype=np.float64)
    if arr.ndim == 3:
        sh, sw = int(arr.shape[0]), int(arr.shape[1])
        y_idx, x_idx = _nearest_resize_index_maps(sh, sw, dst_h, dst_w)
        return arr[y_idx[:, None], x_idx[None, :], :]
    if arr.ndim == 4:
        sh, sw = int(arr.shape[1]), int(arr.shape[2])
        y_idx, x_idx = _nearest_resize_index_maps(sh, sw, dst_h, dst_w)
        return arr[:, y_idx[:, None], x_idx[None, :], :]
    raise ValueError(f"src must be (H,W,C) or (F,H,W,C); got {arr.shape}")


def nearest_resize_adjoint(
    dst_saliency: np.ndarray, src_h: int, src_w: int
) -> np.ndarray:
    """A^T for nearest-resize: scatter-add dest saliency into source cells.

    ``dst_saliency`` is ``(dh, dw, c)`` or ``(F, dh, dw, c)``. Returns the
    source-grid ``(src_h, src_w, c)`` / ``(F, src_h, src_w, c)`` accumulation —
    each source cell receives the SUM of every dest cell that gathered from it
    (block sum-pool). This is the exact transpose of ``nearest_resize_forward``.
    """
    arr = np.asarray(dst_saliency, dtype=np.float64)
    if arr.ndim == 3:
        dh, dw, c = arr.shape
        y_idx, x_idx = _nearest_resize_index_maps(src_h, src_w, dh, dw)
        out = np.zeros((src_h, src_w, c), dtype=np.float64)
        # Vectorized scatter-add over the two axes: build a (dh, dw) -> (sh, sw)
        # incidence and accumulate. np.add.at handles repeated indices correctly.
        yy = np.broadcast_to(y_idx[:, None], (dh, dw)).reshape(-1)
        xx = np.broadcast_to(x_idx[None, :], (dh, dw)).reshape(-1)
        flat = arr.reshape(dh * dw, c)
        np.add.at(out, (yy, xx), flat)
        return out
    if arr.ndim == 4:
        f, dh, dw, c = arr.shape
        y_idx, x_idx = _nearest_resize_index_maps(src_h, src_w, dh, dw)
        out = np.zeros((f, src_h, src_w, c), dtype=np.float64)
        yy = np.broadcast_to(y_idx[:, None], (dh, dw)).reshape(-1)
        xx = np.broadcast_to(x_idx[None, :], (dh, dw)).reshape(-1)
        for fi in range(f):
            flat = arr[fi].reshape(dh * dw, c)
            np.add.at(out[fi], (yy, xx), flat)
        return out
    raise ValueError(f"dst_saliency must be (dh,dw,c) or (F,dh,dw,c); got {arr.shape}")


def _bilinear_index_weight_maps(
    src_h: int, src_w: int, dst_h: int, dst_w: int
) -> dict[str, np.ndarray]:
    """Exact index/weight maps used by ``learned_receiver._bilinear_resize_batch``."""
    y = ((np.arange(dst_h, dtype=np.float64) + 0.5) * (src_h / dst_h)) - 0.5
    x = ((np.arange(dst_w, dtype=np.float64) + 0.5) * (src_w / dst_w)) - 0.5
    y = np.clip(y, 0.0, float(src_h - 1))
    x = np.clip(x, 0.0, float(src_w - 1))
    y0 = np.floor(y).astype(np.int64).clip(0, src_h - 1)
    x0 = np.floor(x).astype(np.int64).clip(0, src_w - 1)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    wy = (y - y0.astype(np.float64))  # (dh,)
    wx = (x - x0.astype(np.float64))  # (dw,)
    return {"y0": y0, "y1": y1, "x0": x0, "x1": x1, "wy": wy, "wx": wx}


def bilinear_resize_forward(src: np.ndarray, dst_h: int, dst_w: int) -> np.ndarray:
    """Forward bilinear-resize gather (mirrors ``_bilinear_resize_batch``).

    ``src`` is ``(F, sh, sw, c)``. align_corners=False, matching the HPRC
    receiver's only supported bilinear alignment.
    """
    arr = np.asarray(src, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError(f"bilinear forward src must be (F,H,W,C); got {arr.shape}")
    sh, sw = int(arr.shape[1]), int(arr.shape[2])
    if sh == dst_h and sw == dst_w:
        return arr.copy()
    m = _bilinear_index_weight_maps(sh, sw, dst_h, dst_w)
    wy = m["wy"].reshape((1, dst_h, 1, 1))
    wx = m["wx"].reshape((1, 1, dst_w, 1))
    top = arr[:, m["y0"], :, :] * (1.0 - wy) + arr[:, m["y1"], :, :] * wy
    out = top[:, :, m["x0"], :] * (1.0 - wx) + top[:, :, m["x1"], :] * wx
    return out


def bilinear_resize_adjoint(
    dst_saliency: np.ndarray, src_h: int, src_w: int
) -> np.ndarray:
    """A^T for bilinear-resize: scatter-add each dest cell's 4 weighted parents.

    Exact transpose of ``bilinear_resize_forward`` for align_corners=False.
    ``dst_saliency`` is ``(F, dh, dw, c)``; returns ``(F, src_h, src_w, c)``.
    """
    arr = np.asarray(dst_saliency, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError(f"bilinear adjoint dst must be (F,dh,dw,c); got {arr.shape}")
    f, dh, dw, c = arr.shape
    if src_h == dh and src_w == dw:
        return arr.copy()
    m = _bilinear_index_weight_maps(src_h, src_w, dh, dw)
    y0, y1, x0, x1, wy, wx = (m["y0"], m["y1"], m["x0"], m["x1"], m["wy"], m["wx"])
    out = np.zeros((f, src_h, src_w, c), dtype=np.float64)
    # forward: out[i,j] = sum over (y in {y0[i],y1[i]}, x in {x0[j],x1[j]}) of
    #   wcoef(y,i)*wcoef(x,j)*src[y,x]. Adjoint scatters dst[i,j] back with the
    #   SAME coefficients. Build the per-(i,j) 4-corner contributions and add.
    wy0 = (1.0 - wy)  # (dh,)
    wy1 = wy
    wx0 = (1.0 - wx)  # (dw,)
    wx1 = wx
    # Per-dest-row weight matrices (dh,) and per-dest-col (dw,). We accumulate
    # by iterating the 4 corners; each corner is a (dest -> source) scatter.
    yy_pairs = ((y0, wy0), (y1, wy1))
    xx_pairs = ((x0, wx0), (x1, wx1))
    for fi in range(f):
        frame = arr[fi]  # (dh, dw, c)
        for y_src, ycoef in yy_pairs:
            for x_src, xcoef in xx_pairs:
                # contribution to source (y_src[i], x_src[j]) is
                #   ycoef[i]*xcoef[j]*frame[i,j,:]
                contrib = frame * (ycoef[:, None, None] * xcoef[None, :, None])
                yy = np.broadcast_to(y_src[:, None], (dh, dw)).reshape(-1)
                xx = np.broadcast_to(x_src[None, :], (dh, dw)).reshape(-1)
                np.add.at(out[fi], (yy, xx), contrib.reshape(dh * dw, c))
    return out


# ---------------------------------------------------------------------------
# Composite residual-stage decode adjoint A_resid^T.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HprcDecodeGeometry:
    """The linear-stage geometry of an HPRC compact-receiver decode.

    Read off ``CompactReceiverPacket`` + ``rdo_plan`` so the adjoint exactly
    mirrors the SHIPPED forward synthesis (no idealized constants).
    """

    decoder_height: int  # decoder-grid H (the "low" synthesis grid)
    decoder_width: int  # decoder-grid W
    camera_height: int  # output resize target H (camera-native)
    camera_width: int  # output resize target W
    residual_grid_h: int  # residual-token grid H (where bits are spent)
    residual_grid_w: int  # residual-token grid W
    channels: int
    residual_scale: float  # int8 token scale
    residual_gain: float  # rdo_plan residual_gain
    output_resize_mode: str  # "nearest" | "bilinear"


def push_pixel_saliency_to_residual_grid(
    pixel_saliency: np.ndarray,
    geometry: HprcDecodeGeometry,
    *,
    selector: np.ndarray | float,
    collapse_channels: bool = True,
) -> np.ndarray:
    """A_resid^T : push a pixel-space saliency surface to the residual-token grid.

    The residual stage of the HPRC decode (for the score-relevant un-clamped
    synthesis) is, per frame ``f``:

        frame_contrib[f] = residual_gain * selector[f] * output_resize(
            residual_scale * nearest_resize(residual_q[f], decoder_grid) )

    A = output_resize ∘ (residual_scale·gain·selector) ∘ nearest_resize. The
    adjoint reverses the composition order with each stage's transpose:

        A^T y = residual_scale·gain·selector · nearest_resize^T(output_resize^T(y))

    ``pixel_saliency`` is a NON-NEGATIVE per-pixel saliency surface
    ``(camera_h, camera_w)`` or ``(camera_h, camera_w, channels)`` or with a
    leading frame dim. Returns the residual-token-grid saliency
    ``(frames, grid_h, grid_w)`` (channels summed if ``collapse_channels``) —
    exactly the ``importance`` shape the ``rate_collapse`` consumer accepts.

    NOTE on the saliency vs gradient distinction: the producer's ``s_seg``/
    ``s_pose`` are SQUARED gradient surfaces (Fisher / flip-risk = grad-energy).
    Energy is not linear, so the formally-exact object is the adjoint of the
    LINEAR decode applied to the (linear) saliency *treated as a co-vector field*
    — i.e. we push the per-pixel IMPORTANCE MASS to the tokens that synthesize
    those pixels, accumulating mass (scatter-add). This is the correct
    coefficient-domain importance for a reverse-waterfill: a token's importance
    = the summed pixel-importance it controls, weighted by the decode gain. The
    EXACTNESS gate (``adjoint_dotproduct_residual``) proves the LINEAR operator's
    adjoint identity; the mass-push is that exact adjoint applied to the
    nonnegative importance field.
    """
    arr = _coerce_pixel_field(pixel_saliency, geometry)  # (F, ch, cam_h, cam_w) -> (F, cam_h, cam_w, C)
    frames = arr.shape[0]
    # Stage 1 adjoint: output_resize^T  (camera -> decoder grid).
    if geometry.output_resize_mode == "nearest":
        dec = nearest_resize_adjoint(arr, geometry.decoder_height, geometry.decoder_width)
    elif geometry.output_resize_mode == "bilinear":
        dec = bilinear_resize_adjoint(arr, geometry.decoder_height, geometry.decoder_width)
    else:
        raise ValueError(f"unsupported output_resize_mode {geometry.output_resize_mode!r}")
    # Stage 2 adjoint: nearest_resize^T (decoder grid -> residual-token grid).
    tok = nearest_resize_adjoint(dec, geometry.residual_grid_h, geometry.residual_grid_w)
    # Stage 3 adjoint: scalar multiplies (scale * gain * selector[f]).
    sel = _coerce_selector(selector, frames)
    gain_scale = float(geometry.residual_scale) * float(geometry.residual_gain)
    tok = tok * (gain_scale * sel.reshape((frames, 1, 1, 1)))
    if collapse_channels:
        return tok.sum(axis=3)  # (F, grid_h, grid_w) — broadcasts over channel in consumer
    return tok  # (F, grid_h, grid_w, channels)


def push_pixel_saliency_to_latent(
    pixel_saliency: np.ndarray,
    *,
    basis: np.ndarray,
    latent_gain: float,
    camera_to_decoder_adjoint_mode: str | None = None,
    geometry: HprcDecodeGeometry | None = None,
) -> np.ndarray:
    """A_latent^T : push a pixel-space saliency surface to the latent dims.

    The latent stage is ``frame_contrib[f] = latent_gain * (latent[f,:] @ basis)``
    where ``basis`` is ``(K, dec_h, dec_w, C)`` on the DECODER grid. So
    ``A_latent[f] = latent_gain * tensordot(latent[f], basis, (0,0))`` is a linear
    map ``R^K -> decoder-grid``. The adjoint ``A_latent^T : decoder-grid -> R^K``
    is ``latent_gain * <basis_k, y>`` summed over spatial+channel — i.e.
    ``latent_gain * (basis_flat @ y_flat)``.

    If ``pixel_saliency`` is at CAMERA resolution, pass ``geometry`` +
    ``camera_to_decoder_adjoint_mode`` so the output-resize adjoint maps it to the
    decoder grid first. Returns ``(frames, K)`` latent-dim saliency.
    """
    basis_f = np.asarray(basis, dtype=np.float64)  # (K, dec_h, dec_w, C)
    if basis_f.ndim != 4:
        raise ValueError(f"basis must be (K,dec_h,dec_w,C); got {basis_f.shape}")
    k, dh, dw, c = basis_f.shape
    arr = np.asarray(pixel_saliency, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, :, :, None]
    elif arr.ndim == 3:
        # (H, W, C) single frame OR (F, H, W) — disambiguate by channel count.
        arr = arr[None, ...] if arr.shape[-1] == c else arr[..., None]
    if arr.ndim != 4:
        raise ValueError(f"pixel_saliency must reduce to (F,H,W,C); got {arr.shape}")
    # Map camera-grid saliency to decoder grid if needed.
    if (arr.shape[1], arr.shape[2]) != (dh, dw):
        if geometry is None or camera_to_decoder_adjoint_mode is None:
            raise ValueError(
                "pixel_saliency not on decoder grid; pass geometry + "
                "camera_to_decoder_adjoint_mode to apply the output-resize adjoint"
            )
        if camera_to_decoder_adjoint_mode == "nearest":
            arr = nearest_resize_adjoint(arr, dh, dw)
        elif camera_to_decoder_adjoint_mode == "bilinear":
            arr = bilinear_resize_adjoint(arr, dh, dw)
        else:
            raise ValueError(
                f"unsupported camera_to_decoder_adjoint_mode {camera_to_decoder_adjoint_mode!r}"
            )
    if arr.shape[-1] == 1 and c > 1:
        arr = np.broadcast_to(arr, (arr.shape[0], dh, dw, c))
    frames = arr.shape[0]
    basis_flat = basis_f.reshape((k, dh * dw * c))  # (K, D)
    y_flat = arr.reshape((frames, dh * dw * c))  # (F, D)
    latent_sal = float(latent_gain) * (y_flat @ basis_flat.T)  # (F, K)
    return latent_sal


# ---------------------------------------------------------------------------
# G3 EXACTNESS GATE — the canonical adjoint dot-product test.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdjointExactnessReport:
    """Result of the canonical ``<A x, y> == <x, A^T y>`` adjoint test."""

    operator: str  # "nearest_resize" | "bilinear_resize" | "residual_decode" | "latent_decode"
    lhs_inner: float  # <A x, y>_Y  (pixel-domain inner product)
    rhs_inner: float  # <x, A^T y>_X (token/latent-domain inner product)
    abs_residual: float  # |lhs - rhs|
    rel_residual: float  # |lhs - rhs| / (|lhs| + eps)
    is_exact: bool  # rel_residual <= tol
    tol: float
    src_shape: tuple[int, ...]
    dst_shape: tuple[int, ...]

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "lhs_inner_Ax_y": self.lhs_inner,
            "rhs_inner_x_ATy": self.rhs_inner,
            "abs_residual": self.abs_residual,
            "rel_residual": self.rel_residual,
            "is_exact": self.is_exact,
            "tol": self.tol,
            "src_shape": list(self.src_shape),
            "dst_shape": list(self.dst_shape),
        }


def _adjoint_dotproduct(
    *,
    operator: str,
    x: np.ndarray,
    y: np.ndarray,
    forward,
    adjoint,
    tol: float,
) -> AdjointExactnessReport:
    """Canonical adjoint identity check ``<A x, y> = <x, A^T y>``.

    ``x`` lives in the SOURCE (token/coefficient) domain; ``y`` lives in the
    DEST (pixel) domain. ``forward(x) = A x`` (source->dest), ``adjoint(y) = A^T y``
    (dest->source). The adjoint is exact iff ``<A x, y> == <x, A^T y>`` for ALL
    x, y — we check it on random tensors to numerical tolerance. A transform
    that is NOT the true transpose fails this (NO-FAKE guard).
    """
    ax = np.asarray(forward(x), dtype=np.float64)
    aty = np.asarray(adjoint(y), dtype=np.float64)
    lhs = float(np.sum(ax * np.asarray(y, dtype=np.float64)))
    rhs = float(np.sum(np.asarray(x, dtype=np.float64) * aty))
    abs_res = abs(lhs - rhs)
    rel_res = abs_res / (abs(lhs) + 1e-300)
    return AdjointExactnessReport(
        operator=operator,
        lhs_inner=lhs,
        rhs_inner=rhs,
        abs_residual=abs_res,
        rel_residual=rel_res,
        is_exact=rel_res <= tol,
        tol=tol,
        src_shape=tuple(np.asarray(x).shape),
        dst_shape=tuple(np.asarray(y).shape),
    )


def adjoint_dotproduct_nearest(
    *,
    src_h: int,
    src_w: int,
    dst_h: int,
    dst_w: int,
    channels: int = 3,
    seed: int = 0,
    tol: float = 1e-9,
) -> AdjointExactnessReport:
    """Dot-product exactness test for the nearest-resize adjoint."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((src_h, src_w, channels))  # source (token grid)
    y = rng.standard_normal((dst_h, dst_w, channels))  # dest (pixel grid)
    return _adjoint_dotproduct(
        operator="nearest_resize",
        x=x,
        y=y,
        forward=lambda v: nearest_resize_forward(v, dst_h, dst_w),
        adjoint=lambda v: nearest_resize_adjoint(v, src_h, src_w),
        tol=tol,
    )


def adjoint_dotproduct_bilinear(
    *,
    src_h: int,
    src_w: int,
    dst_h: int,
    dst_w: int,
    channels: int = 3,
    frames: int = 1,
    seed: int = 0,
    tol: float = 1e-9,
) -> AdjointExactnessReport:
    """Dot-product exactness test for the bilinear-resize adjoint."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((frames, src_h, src_w, channels))
    y = rng.standard_normal((frames, dst_h, dst_w, channels))
    return _adjoint_dotproduct(
        operator="bilinear_resize",
        x=x,
        y=y,
        forward=lambda v: bilinear_resize_forward(v, dst_h, dst_w),
        adjoint=lambda v: bilinear_resize_adjoint(v, src_h, src_w),
        tol=tol,
    )


def adjoint_dotproduct_residual(
    geometry: HprcDecodeGeometry,
    *,
    selector: np.ndarray | float = 1.0,
    frames: int = 2,
    seed: int = 0,
    tol: float = 1e-9,
) -> AdjointExactnessReport:
    """Dot-product exactness test for the FULL composite residual-decode adjoint.

    Builds A x in the pixel domain (residual_q -> camera frame) via the same
    forward stages the HPRC receiver ships, and A^T y in the token domain via
    ``push_pixel_saliency_to_residual_grid``, then checks ``<A x, y> = <x, A^T y>``.
    The channel-collapse path is NOT used here (it changes the operator); the
    exactness test uses the per-channel adjoint.
    """
    rng = np.random.default_rng(seed)
    g = geometry
    sel = _coerce_selector(selector, frames)
    gain_scale = float(g.residual_scale) * float(g.residual_gain)

    def forward(x_tok: np.ndarray) -> np.ndarray:
        # x_tok: (F, grid_h, grid_w, C) residual tokens (in token units, pre-scale)
        dec = nearest_resize_forward(x_tok, g.decoder_height, g.decoder_width)
        dec = dec * (gain_scale * sel.reshape((frames, 1, 1, 1)))
        if g.output_resize_mode == "nearest":
            return nearest_resize_forward(dec, g.camera_height, g.camera_width)
        return bilinear_resize_forward(dec, g.camera_height, g.camera_width)

    def adjoint(y_pix: np.ndarray) -> np.ndarray:
        return push_pixel_saliency_to_residual_grid(
            y_pix, g, selector=selector, collapse_channels=False
        )

    x = rng.standard_normal(
        (frames, g.residual_grid_h, g.residual_grid_w, g.channels)
    )
    y = rng.standard_normal((frames, g.camera_height, g.camera_width, g.channels))
    return _adjoint_dotproduct(
        operator="residual_decode",
        x=x,
        y=y,
        forward=forward,
        adjoint=adjoint,
        tol=tol,
    )


def adjoint_dotproduct_latent(
    *,
    basis: np.ndarray,
    latent_gain: float = 1.0,
    frames: int = 2,
    seed: int = 0,
    tol: float = 1e-9,
) -> AdjointExactnessReport:
    """Dot-product exactness test for the latent-decode adjoint (on decoder grid)."""
    rng = np.random.default_rng(seed)
    basis_f = np.asarray(basis, dtype=np.float64)
    k, dh, dw, c = basis_f.shape
    basis_flat = basis_f.reshape((k, dh * dw * c))

    def forward(x_lat: np.ndarray) -> np.ndarray:
        # x_lat: (F, K) -> decoder grid (F, dh, dw, c)
        out = float(latent_gain) * (x_lat @ basis_flat)  # (F, D)
        return out.reshape((x_lat.shape[0], dh, dw, c))

    def adjoint(y_pix: np.ndarray) -> np.ndarray:
        return push_pixel_saliency_to_latent(
            y_pix, basis=basis_f, latent_gain=latent_gain
        )

    x = rng.standard_normal((frames, k))
    y = rng.standard_normal((frames, dh, dw, c))
    return _adjoint_dotproduct(
        operator="latent_decode",
        x=x,
        y=y,
        forward=forward,
        adjoint=adjoint,
        tol=tol,
    )


# ---------------------------------------------------------------------------
# Geometry extraction from a live HPRC compact-receiver packet.
# ---------------------------------------------------------------------------


def geometry_from_compact_packet(
    compact: Any,
    *,
    camera_height: int,
    camera_width: int,
) -> HprcDecodeGeometry:
    """Read the decode geometry off a ``CompactReceiverPacket`` + rdo_plan.

    No idealized constants — every field is the SHIPPED forward op's parameter.
    """
    decoder = compact.decoder
    rdo = compact.rdo_plan
    residual = compact.residual
    mode = str(rdo.get("output_resize", "nearest"))
    return HprcDecodeGeometry(
        decoder_height=int(decoder.height),
        decoder_width=int(decoder.width),
        camera_height=int(camera_height),
        camera_width=int(camera_width),
        residual_grid_h=int(residual.grid_h),
        residual_grid_w=int(residual.grid_w),
        channels=int(residual.channels),
        residual_scale=float(residual.scale),
        residual_gain=float(rdo.get("residual_gain", 1.0)),
        output_resize_mode=mode,
    )


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------


def _coerce_pixel_field(
    pixel_saliency: np.ndarray, geometry: HprcDecodeGeometry
) -> np.ndarray:
    """Coerce a pixel saliency surface to (F, cam_h, cam_w, C)."""
    arr = np.asarray(pixel_saliency, dtype=np.float64)
    ch = geometry.channels
    if arr.ndim == 2:  # (cam_h, cam_w) -> 1 frame, broadcast over channels
        return np.broadcast_to(arr[None, :, :, None], (1, arr.shape[0], arr.shape[1], ch)).copy()
    if arr.ndim == 3:
        if arr.shape[-1] == ch:  # (cam_h, cam_w, C)
            return arr[None, ...]
        # (F, cam_h, cam_w) -> broadcast over channels
        return np.broadcast_to(arr[..., None], (*arr.shape, ch)).copy()
    if arr.ndim == 4:  # (F, cam_h, cam_w, C)
        if arr.shape[-1] == 1 and ch > 1:
            return np.broadcast_to(arr, (*arr.shape[:3], ch)).copy()
        return arr
    raise ValueError(
        f"pixel_saliency must be (H,W) / (H,W,C) / (F,H,W) / (F,H,W,C); got {arr.shape}"
    )


def _coerce_selector(selector: np.ndarray | float, frames: int) -> np.ndarray:
    if np.isscalar(selector):
        return np.full((frames,), float(selector), dtype=np.float64)
    sel = np.asarray(selector, dtype=np.float64).reshape(-1)
    if sel.shape[0] == frames:
        return sel
    if sel.shape[0] == 1:
        return np.full((frames,), float(sel[0]), dtype=np.float64)
    raise ValueError(f"selector length {sel.shape[0]} != frames {frames}")


__all__ = [
    "AdjointExactnessReport",
    "HprcDecodeGeometry",
    "adjoint_dotproduct_bilinear",
    "adjoint_dotproduct_latent",
    "adjoint_dotproduct_nearest",
    "adjoint_dotproduct_residual",
    "bilinear_resize_adjoint",
    "bilinear_resize_forward",
    "geometry_from_compact_packet",
    "nearest_resize_adjoint",
    "nearest_resize_forward",
    "push_pixel_saliency_to_latent",
    "push_pixel_saliency_to_residual_grid",
]
