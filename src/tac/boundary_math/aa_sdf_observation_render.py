"""AA-SDF observation-map render — anti-aliased footprint-integrated witness render.

THE MEASURED #1 REPRESENTATION LEVER (``tools/levelset_gate_discriminators_n600.py``,
DAG FEED-ly, ``[macOS-CPU advisory] NON-PROMOTABLE``): POINT-sampling the witness render
grid ERASES finest-scale lane structure (the class-1 dashes); FOOTPRINT-INTEGRATED
(supersample->box or mip-NeRF IPE cone) rendering RECOVERS it. Measured through the ACTUAL
contest R (``apply_contest_faithful_roundtrip_nhwc``) + the frozen CPU-torch SegNet argmax:

* oracle-R d_seg floor 0.00091 @ render 384x512 (area/AA) vs 0.00247 @ 192x256 (point);
* class-1 (LANE) recall +0.38 (0.56 point-sampled -> 0.94 supersampled) at the 192 grid.

This is a ~0-RATE lever: the render is a DETERMINISTIC decode-time operation (inflate.py runs
it for free within the 30-min budget per CLAUDE.md rule 118); the witness weights + codes in
``archive.zip`` are UNCHANGED. So AA moves d_seg without moving the rate term.

Two AA modes over the WHOLE softmax-of-SDF partition (the LevelSetRGBWitness ``_compose_rgb``
output), composited BEFORE R:

  1. **SUPERSAMPLE -> BOX** (ground-truth pixel-footprint integration). Render the witness at
     ``(ss*H, ss*W)`` then AREA/box-downsample to ``(H, W)``. For an INTEGER factor ``ss`` the
     area filter is EXACTLY the ss x ss block mean (bit-identical to
     ``torch.nn.functional.interpolate(mode="area")`` for integer factors, which is what the
     committed gate's ``area_down`` uses). Fully GENERAL: works on any render field (RGB INR,
     softmax-of-SDF palette+texture, ...). Cost: ss^2 x the witness forward.

  2. **mip-NeRF IPE CONE** (analytical, ~0 extra compute; Barron et al. mip-NeRF 2103.13415).
     Instead of supersampling, integrate the positional encoding over the per-pixel Gaussian
     footprint. For the curvelet basis ``feats = [sin(2*pi coords@B), cos(2*pi coords@B)]`` the
     footprint expectation is ``E[sin(2*pi b.x)] = sin(2*pi b.mu) * exp(-2*pi^2 b^T Sigma b)``
     (same exp factor for the cos twin). So each Fourier column is multiplied by
     ``att_c = exp(-2*pi^2 (Bx_c^2 sx^2 + By_c^2 sy^2))`` where ``(sx, sy)`` is the footprint
     std per axis. This anti-aliases the BASIS (attenuates high frequencies that alias below
     the render Nyquist) with NO supersampling. It is an ANALYTICAL APPROXIMATION to (1): it
     integrates the INPUT features, not the (nonlinear) RGB output, so (1) is the authority and
     (2) is the cheap decode-time proxy. Both are valid ~0-rate observation models.

THE ``--render-grid`` LEVER (third knob) is the base grid ``(H, W)`` the R operator sees: the
oracle-R floor drops monotonically as ``(H, W)`` approaches camera res (874x1164), because R
bicubic-UPsamples the render grid to camera. The LEVELSET witness already defaults to 384x512
(``--render-h 384 --render-w 512``); raising it further (512, 768, ...) trades forward compute
for a lower achievable-through-R floor. AA (supersample/IPE) is COMPLEMENTARY: it recovers the
finest scale WITHIN a fixed render grid.

DISCIPLINE: MLX-first (``mx``) + numpy-portable bit-identical reference. The box-downsample is
an exact reshape-mean (numpy is the authority; MLX matches to fp tolerance, parity >= 0.9997);
the IPE attenuation is a deterministic per-column multiplier (pure numpy). Composes with
``apply_contest_faithful_roundtrip_nhwc`` (the fused-R; DAG bug-#1 contest-exact) and
``render_batch_through_R_mlx``. NEVER an authority score — the frozen CPU-torch numpy-fp32
SegNet on the exact bytes is the only verdict (MPS never).

Canonical equation: ``aa_sdf_observation_footprint_render_dseg_v1``.
DSL gauge SPEC: ``--render-aa {none,supersample,ipe}`` / ``--aa-supersample S`` / render-grid.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# The frozen contest scorer input resolution (== ``SCORER_HW`` in
# ``tac.local_acceleration.pr95_hnerv_mlx_training``; SegNet.preprocess_input bilinear-downsamples
# the camera-res uint8 frame to this float grid). Kept local so this module has NO heavy import at
# module load; a test asserts equality against the canonical constant.
SEG_H, SEG_W = 384, 512

# Render coord grid span. Mirrors ``experiments/train_witness_realized_through_R_mlx.py::
# _build_render_coords`` EXACTLY (linspace(-1, 1, n) per axis, endpoint-inclusive). A test asserts
# bit-identity against the trainer helper so ss=1 rendering is byte-identical to the current path.
RENDER_COORD_LO, RENDER_COORD_HI = -1.0, 1.0


class AASDFRenderError(ValueError):
    """Raised on malformed AA-SDF observation-render arguments (fail-closed)."""


# ---------------------------------------------------------------------------
# Coord grids (numpy authority). ss=1 is bit-identical to the trainer's _build_render_coords.
# ---------------------------------------------------------------------------
def build_render_coords(h: int, w: int) -> np.ndarray:
    """``(h*w, 2)`` float32 render coords, row-major, columns ``[x, y]`` in ``[-1, 1]``.

    Bit-identical to ``experiments/train_witness_realized_through_R_mlx.py::_build_render_coords``:
    ``ys = linspace(-1,1,h)``, ``xs = linspace(-1,1,w)``, ``meshgrid(ys, xs, indexing='ij')``,
    ``stack([gx.ravel(), gy.ravel()], -1)``. Pixel ``(row i, col j)`` -> coord ``(xs[j], ys[i])``.
    """
    if h < 1 or w < 1:
        raise AASDFRenderError(f"render grid must be positive; got {(h, w)}")
    ys = np.linspace(RENDER_COORD_LO, RENDER_COORD_HI, h, dtype=np.float32)
    xs = np.linspace(RENDER_COORD_LO, RENDER_COORD_HI, w, dtype=np.float32)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)


def build_supersampled_coords(render_h: int, render_w: int, ss: int) -> np.ndarray:
    """Coords at the SUPERSAMPLED grid ``(ss*render_h, ss*render_w)`` -> ``(ss^2*H*W, 2)`` float32.

    ``ss == 1`` returns EXACTLY ``build_render_coords(render_h, render_w)`` (bit-identical => the
    downstream witness render + R is byte-identical to the current point-sampled path). The fine
    grid spans the SAME ``[-1, 1]`` box, so box-downsampling ss x ss fine samples integrates each
    coarse pixel's footprint (standard supersampling; matches torch ``area`` for integer factors).
    """
    ss = _check_ss(ss)
    return build_render_coords(render_h * ss, render_w * ss)


def _check_ss(ss: int) -> int:
    ss_int = int(ss)
    if ss_int < 1:
        raise AASDFRenderError(f"aa-supersample factor ss must be >= 1; got {ss}")
    return ss_int


# ---------------------------------------------------------------------------
# Box (area) downsample — the ground-truth footprint integrator. numpy authority + MLX twin.
# Integer factor => EXACT ss x ss block mean == torch interpolate(mode='area') for integer factors.
# ---------------------------------------------------------------------------
def box_downsample_np(x_nhwc: np.ndarray, ss: int) -> np.ndarray:
    """``(N, ss*H, ss*W, C) -> (N, H, W, C)`` by ss x ss block mean (numpy authority)."""
    ss = _check_ss(ss)
    x = np.asarray(x_nhwc)
    if x.ndim != 4:
        raise AASDFRenderError(f"box_downsample expects NHWC (4D); got shape {x.shape}")
    n, fh, fw, c = x.shape
    if ss == 1:
        return x
    if fh % ss or fw % ss:
        raise AASDFRenderError(
            f"fine grid {(fh, fw)} not divisible by ss={ss} (expected ss*(H,W))"
        )
    h, w = fh // ss, fw // ss
    return x.reshape(n, h, ss, w, ss, c).mean(axis=(2, 4))


def box_downsample_mlx(x_nhwc: Any, ss: int) -> Any:
    """MLX twin of :func:`box_downsample_np` (``(N, ss*H, ss*W, C) -> (N, H, W, C)``).

    Bit-identical STRUCTURE (reshape + mean over the ss axes); values match the numpy authority to
    fp tolerance (a mean of ss^2 finite floats; parity >> 0.9997 per the MLX portability contract).
    """
    import mlx.core as mx

    ss = _check_ss(ss)
    if x_nhwc.ndim != 4:
        raise AASDFRenderError(f"box_downsample expects NHWC (4D); got shape {x_nhwc.shape}")
    n, fh, fw, c = (int(d) for d in x_nhwc.shape)
    if ss == 1:
        return x_nhwc
    if fh % ss or fw % ss:
        raise AASDFRenderError(
            f"fine grid {(fh, fw)} not divisible by ss={ss} (expected ss*(H,W))"
        )
    h, w = fh // ss, fw // ss
    resh = mx.reshape(x_nhwc, (n, h, ss, w, ss, c))
    return mx.mean(resh, axis=(2, 4))


def box_downsample(x_nhwc: Any, ss: int) -> Any:
    """Dispatch box-downsample by array type (numpy -> numpy authority; else MLX twin)."""
    if isinstance(x_nhwc, np.ndarray):
        return box_downsample_np(x_nhwc, ss)
    return box_downsample_mlx(x_nhwc, ss)


# ---------------------------------------------------------------------------
# mip-NeRF IPE cone — analytical Fourier-feature footprint attenuation (pure numpy, deterministic).
# ---------------------------------------------------------------------------
def ipe_footprint_sigma(
    render_h: int, render_w: int, footprint_scale: float = 1.0
) -> tuple[float, float]:
    """Per-axis footprint std ``(sx, sy)`` in coord units for one render pixel.

    Coord span is 2.0 over ``n`` samples -> pixel pitch ``d = 2/n`` per axis. A box footprint of
    width ``d`` has variance ``d^2/12`` -> ``sigma = d/sqrt(12)``. ``footprint_scale`` (default 1.0
    = one-pixel box) tunes the cone aperture. Larger => stronger high-frequency attenuation.
    """
    if render_h < 1 or render_w < 1:
        raise AASDFRenderError(f"render grid must be positive; got {(render_h, render_w)}")
    if footprint_scale <= 0.0:
        raise AASDFRenderError(f"footprint_scale must be > 0; got {footprint_scale}")
    inv_sqrt12 = 1.0 / np.sqrt(12.0)
    sx = float(footprint_scale) * (2.0 / render_w) * inv_sqrt12
    sy = float(footprint_scale) * (2.0 / render_h) * inv_sqrt12
    return sx, sy


def ipe_curvelet_attenuation(
    B: np.ndarray, sigma_x: float, sigma_y: float
) -> np.ndarray:
    """Per-Fourier-column IPE attenuation ``(cols,)`` for a curvelet basis ``B`` of shape ``(2, cols)``.

    ``B[0]`` = x-frequencies, ``B[1]`` = y-frequencies (cycles per coord unit; the ``curvelet_feats``
    projection is ``2*pi * coords@B``). For a Gaussian footprint ``N(mu, diag(sx^2, sy^2))`` the
    positional-encoding expectation attenuates every column by
    ``att_c = exp(-2*pi^2 (Bx_c^2 sx^2 + By_c^2 sy^2))`` (mip-NeRF IPE; same factor for sin & cos).
    """
    Bx = np.asarray(B, np.float64)
    if Bx.ndim != 2 or Bx.shape[0] != 2:
        raise AASDFRenderError(f"curvelet B must be (2, cols); got {Bx.shape}")
    two_pi2 = 2.0 * np.pi * np.pi
    quad = (Bx[0] ** 2) * (float(sigma_x) ** 2) + (Bx[1] ** 2) * (float(sigma_y) ** 2)
    return np.exp(-two_pi2 * quad).astype(np.float32)


def apply_ipe_attenuation(curv_feats: np.ndarray, att: np.ndarray) -> np.ndarray:
    """Attenuate ``[sin, cos]`` curvelet feats ``(P, 2*cols)`` by the per-column IPE factor ``att``.

    ``curvelet_feats`` returns ``concatenate([sin(proj), cos(proj)], -1)``; both halves share the
    per-frequency ``att`` (the exp footprint factor is identical for the sin and cos twins).
    """
    feats = np.asarray(curv_feats)
    cols = feats.shape[-1] // 2
    a = np.asarray(att, feats.dtype)
    if a.shape[-1] != cols:
        raise AASDFRenderError(
            f"att has {a.shape[-1]} cols but feats has {cols} freq cols"
        )
    tiled = np.concatenate([a, a], axis=-1)
    return (feats * tiled).astype(feats.dtype)


# ---------------------------------------------------------------------------
# AA render-through-R (MLX, differentiable) — drop-in twins of render_batch_through_R_mlx /
# render_through_R_mlx with footprint integration composited BEFORE the contest-exact R.
# ---------------------------------------------------------------------------
def render_aa_batch_through_R_mlx(
    witness: Any,
    coord_feats_fine: Any,
    code_indices: Any,
    render_h: int,
    render_w: int,
    ss: int,
    *,
    output_hw: tuple[int, int] = (SEG_H, SEG_W),
    compose_fn: Any = None,
) -> Any:
    """SUPERSAMPLE->BOX AA batched render-through-R: ``(M,) codes -> (M, SEG_H, SEG_W, 3)``.

    ``coord_feats_fine`` MUST be the coord feats at the SUPERSAMPLED grid ``(ss*render_h,
    ss*render_w)`` (i.e. ``curvelet_feats(build_supersampled_coords(render_h, render_w, ss), B)``
    plus any per-pair directional augmentation, exactly as the trainer builds ``coord_feats_mx``
    but at the fine grid). ``(render_h, render_w)`` is the BASE grid R sees after box-downsample.

    Pipeline: witness.call_batch (fine grid) -> reshape (M, ss*H, ss*W, 3) -> optional
    ``compose_fn`` (residual bulk compose, BEFORE downsample) -> box-downsample ss -> (M, H, W, 3)
    -> ``apply_contest_faithful_roundtrip_nhwc`` (bicubic^ to camera -> uint8 @ camera -> bilinear
    down to scorer). ``ss == 1`` is byte-identical to ``render_batch_through_R_mlx`` (no downsample).
    """
    import mlx.core as mx

    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        apply_contest_faithful_roundtrip_nhwc,
    )

    ss = _check_ss(ss)
    fh, fw = render_h * ss, render_w * ss
    rgb_flat = witness.call_batch(coord_feats_fine, code_indices)  # (M, fh*fw, 3)
    rgb = mx.reshape(rgb_flat, (-1, fh, fw, 3))  # (M, ss*H, ss*W, 3) NHWC
    if compose_fn is not None:
        rgb = compose_fn(rgb, code_indices)  # residual bulk compose at the fine grid
    rgb = box_downsample_mlx(rgb, ss)  # (M, H, W, 3) footprint-integrated
    return apply_contest_faithful_roundtrip_nhwc(rgb, output_hw=output_hw, ste_round=True)


def render_aa_through_R_mlx(
    witness: Any,
    coord_feats_fine: Any,
    code_idx: int,
    render_h: int,
    render_w: int,
    ss: int,
    *,
    output_hw: tuple[int, int] = (SEG_H, SEG_W),
    compose_fn: Any = None,
) -> Any:
    """Per-frame SUPERSAMPLE->BOX AA render-through-R -> ``(1, SEG_H, SEG_W, 3)``.

    Single-code twin of :func:`render_aa_batch_through_R_mlx` (drop-in for ``render_through_R_mlx``).
    """
    import mlx.core as mx

    from tac.local_acceleration.pr95_hnerv_mlx_training import (
        apply_contest_faithful_roundtrip_nhwc,
    )

    ss = _check_ss(ss)
    fh, fw = render_h * ss, render_w * ss
    rgb_flat = witness(coord_feats_fine, code_idx)  # (fh*fw, 3)
    rgb = mx.reshape(rgb_flat, (1, fh, fw, 3))
    if compose_fn is not None:
        rgb = compose_fn(rgb, code_idx)
    rgb = box_downsample_mlx(rgb, ss)  # (1, H, W, 3)
    return apply_contest_faithful_roundtrip_nhwc(rgb, output_hw=output_hw, ste_round=True)


__all__ = [
    "SEG_H",
    "SEG_W",
    "AASDFRenderError",
    "build_render_coords",
    "build_supersampled_coords",
    "box_downsample",
    "box_downsample_np",
    "box_downsample_mlx",
    "ipe_footprint_sigma",
    "ipe_curvelet_attenuation",
    "apply_ipe_attenuation",
    "render_aa_batch_through_R_mlx",
    "render_aa_through_R_mlx",
]
