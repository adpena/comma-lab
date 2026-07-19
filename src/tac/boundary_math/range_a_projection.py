# SPDX-License-Identifier: MIT
"""range_a_projection — the EXACT ``P_range(A)`` render-target projector (#520, SPEC_v10 P3).

THE COMPOSITION LAW (docstring, binding — SPEC_v10 §3 P3 amendment 2026-07-17):

    ``P_range(A)`` and the Laguerre cell-generator description are the SAME projection onto
    the frozen scorers' sigma-algebra.  The shared bilinear resize ``A`` (camera
    ``(874,1164) -> (384,512)``, ``align_corners=False``, ``antialias=False`` — the SAME
    kernel on EVERY scored path, upstream ``modules.py:109 == :73``) defines the finest
    measurable partition of camera space: ``ker(A)`` is what the scorer cannot RESOLVE, and
    within a resolved atom the argmax cannot DISTINGUISH constant offsets.  So a render
    target is "cells in ``range(A)``", not pixels.  ``P_range(A)`` removes the ``ker(A)``
    complement (the measured ~52% scorer-invisible render energy, #519); the cell-generator
    description removes the intra-atom indistinguishable degrees of freedom.  They COMPOSE —
    both are projections onto atoms of the same sigma-algebra.

MECHANISM.  ``A`` is separable: ``A(X) = Dr @ X @ Dc^T`` with ``Dr`` ``(384,874)`` and
``Dc`` ``(512,1164)`` the EXACT torch bilinear downsample as row/col matrices (impulse-probed
via the #391 ``resize_matrix_1d`` machinery — NOT re-derived from the cubic-convolution
formula).  The orthogonal projector onto ``ker(A)^perp = range(A^T)`` is separable too:

    ``P(X) = Qr (Qr^T X Qc) Qc^T``

where ``Qr`` is an orthonormal basis of ``range(Dr^T)`` (``874x384``) and ``Qc`` of
``range(Dc^T)`` (``1164x512``).  Then ``X - P(X) in ker(A)`` exactly, so ``A(X - P X) = 0``
(the render's scorer-invisible complement) and ``A(P X) = A(X)`` (the scorer sees the same
thing).  Projector self-test residual: ``max|A(X - P X)| = 1.65e-15`` — the SAME validated
number as #519 (``null_subspace_rate_measure``), reproduced here from the exact operator.

THREE FORMS (SPEC_v10 P3 deliverable, #520):
  * ``apply_projection``      — numpy fp32/fp64 EXACT (THE AUTHORITY).
  * ``apply_projection_mlx``  — the MLX twin, parity-tested vs the numpy authority.
  * ``RangeAProjection``      — the DSL Lever factory (in ``curriculum_dsl``; default OFF,
                                duty-to-measure) that arms a trainer-side render-target
                                projection.  ``maybe_project_render_target`` is the
                                guarded, default-OFF, fail-closed consumption hook.

Axis honesty: this module is pure geometry (the exact resize kernel).  It carries NO score
claim.  ``P_range(A)`` applied to the SCORER INPUT is scorer-invariant BY CONSTRUCTION
(``A(PX)=A(X)``) — its score-moving use is as a TRAINING-RENDER restriction (do not spend
witness capacity on ``ker(A)``); that EFFECT is measurement-gated (SPEC_v10 gate chain).
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

# Camera + scorer grids (upstream/modules.py preprocess resize; pinned).
CAMERA_H, CAMERA_W = 874, 1164
SEG_H, SEG_W = 384, 512

# The validated #519 projector self-test residual (max|A(X-PX)|), reproduced by this module.
REFERENCE_KER_RESIDUAL: float = 1.654926196081874e-15
# Loose ceiling for the exactness assertion (fp64 QR round-off headroom).
KER_RESIDUAL_CEILING: float = 1e-11

_VALID_CADENCES: tuple[str, ...] = ("post_render", "every_step")


class RangeAProjectionError(ValueError):
    """Raised (fail-closed, never silent) when the projector cannot be applied honestly."""


# ---------------------------------------------------------------------------
# Exact separable operator (reuses the #391 resize-kernel machinery; cached).
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _resize_matrices() -> tuple[np.ndarray, np.ndarray]:
    """``Dr (384,874)``, ``Dc (512,1164)``: the EXACT torch bilinear ``align_corners=False``
    ``antialias=False`` downsample as separable row/col matrices (impulse-probed, fp64)."""
    from tac.through_r.flip_inverse import resize_matrix_1d

    dr = resize_matrix_1d(CAMERA_H, SEG_H, "bilinear",
                          align_corners=False, antialias=False, dtype=np.float64)
    dc = resize_matrix_1d(CAMERA_W, SEG_W, "bilinear",
                          align_corners=False, antialias=False, dtype=np.float64)
    return np.ascontiguousarray(dr), np.ascontiguousarray(dc)


@functools.lru_cache(maxsize=1)
def _ortho_bases_f64() -> tuple[np.ndarray, np.ndarray]:
    """``Qr (874,384)``, ``Qc (1164,512)``: orthonormal bases of ``range(Dr^T)`` /
    ``range(Dc^T)`` so ``P = Qr Qr^T (x) Qc Qc^T`` projects onto ``ker(A)^perp`` (fp64)."""
    dr, dc = _resize_matrices()
    qr, rr = np.linalg.qr(dr.T)  # dr.T is (874,384)
    qc, rc = np.linalg.qr(dc.T)  # dc.T is (1164,512)
    if float(np.abs(np.diag(rr)).min()) <= 0.0 or float(np.abs(np.diag(rc)).min()) <= 0.0:
        raise RangeAProjectionError(
            "resize matrix not full row rank — the range(A) projector would be invalid")
    return np.ascontiguousarray(qr), np.ascontiguousarray(qc)


@functools.lru_cache(maxsize=2)
def _ortho_bases(dtype: Any = np.float64) -> tuple[np.ndarray, np.ndarray]:
    qr, qc = _ortho_bases_f64()
    return qr.astype(dtype), qc.astype(dtype)


def _project_stack(stack: np.ndarray, qr: np.ndarray, qc: np.ndarray) -> np.ndarray:
    """Project a ``(B, 874, 1164)`` stack: ``out = Qr (Qr^T X Qc) Qc^T`` per image (2 einsums,
    through the reduced ``(B,384,512)`` space — the cheap round-trip form)."""
    red = np.einsum("hr,bhw,wc->brc", qr, stack, qc, optimize=True)   # (B,384,512)
    out = np.einsum("hr,brc,wc->bhw", qr, red, qc, optimize=True)     # (B,874,1164)
    return out


def _split_layout(x: np.ndarray) -> tuple[np.ndarray, tuple[int, ...], bool, int]:
    """Return ``(flat_BHW, lead_shape, has_channel, n_channel)`` for a camera-shaped array.

    Accepts ``(...,874,1164)`` (no channel) or ``(...,874,1164,C)`` (trailing channel).
    """
    if x.ndim >= 2 and x.shape[-2:] == (CAMERA_H, CAMERA_W):
        lead = x.shape[:-2]
        flat = np.ascontiguousarray(x.reshape(-1, CAMERA_H, CAMERA_W))
        return flat, lead, False, 1
    if x.ndim >= 3 and x.shape[-3:-1] == (CAMERA_H, CAMERA_W):
        c = int(x.shape[-1])
        lead = x.shape[:-3]
        # (...,H,W,C) -> (B,C,H,W) -> (B*C,H,W)
        moved = np.moveaxis(x.reshape(-1, CAMERA_H, CAMERA_W, c), -1, 1)
        flat = np.ascontiguousarray(moved.reshape(-1, CAMERA_H, CAMERA_W))
        return flat, lead, True, c
    raise RangeAProjectionError(
        f"range(A) projection needs camera spatial dims {(CAMERA_H, CAMERA_W)} as the last two "
        f"axes (optionally + a trailing channel); got shape {tuple(x.shape)}")


def apply_projection(frames: Any, *,
                     out_dtype: Any = np.float32,
                     compute_dtype: Any = np.float64) -> np.ndarray:
    """THE AUTHORITY. Project camera-space ``frames`` onto ``range(A)`` (drop ``ker(A)``).

    ``frames`` : ``(874,1164)`` | ``(874,1164,C)`` | ``(...,874,1164[,C])`` array-like.
    Computed in ``compute_dtype`` (fp64 by default → EXACT to the #519 residual), returned in
    ``out_dtype`` (fp32 by default → matches the render pipeline).  Shape preserved.
    """
    a = np.asarray(frames)
    x = a.astype(compute_dtype)
    qr, qc = _ortho_bases(np.dtype(compute_dtype).type)
    flat, lead, has_channel, c = _split_layout(x)
    proj = _project_stack(flat, qr, qc)
    if has_channel:
        proj = np.moveaxis(proj.reshape(-1, c, CAMERA_H, CAMERA_W), 1, -1)  # (B,H,W,C)
        out = proj.reshape(*lead, CAMERA_H, CAMERA_W, c)
    else:
        out = proj.reshape(*lead, CAMERA_H, CAMERA_W)
    return np.ascontiguousarray(out.astype(out_dtype))


def apply_projection_mlx(frames: Any):
    """The MLX twin (fp32).  Same separable projection, parity-tested vs the numpy authority.

    Returns an ``mx.array``.  Import is lazy so the module has no hard MLX dependency.
    """
    import mlx.core as mx

    qr_np, qc_np = _ortho_bases(np.float32)
    qr = mx.array(np.ascontiguousarray(qr_np))
    qc = mx.array(np.ascontiguousarray(qc_np))
    x = frames.astype(mx.float32) if isinstance(frames, mx.array) \
        else mx.array(np.asarray(frames, dtype=np.float32))
    shape = tuple(int(s) for s in x.shape)
    if shape[-2:] == (CAMERA_H, CAMERA_W):
        lead, c, has_channel = shape[:-2], 1, False
        flat = x.reshape(-1, CAMERA_H, CAMERA_W)
    elif len(shape) >= 3 and shape[-3:-1] == (CAMERA_H, CAMERA_W):
        c, has_channel = int(shape[-1]), True
        lead = shape[:-3]
        flat = mx.moveaxis(x.reshape(-1, CAMERA_H, CAMERA_W, c), -1, 1).reshape(
            -1, CAMERA_H, CAMERA_W)
    else:
        raise RangeAProjectionError(
            f"range(A) MLX projection needs camera dims {(CAMERA_H, CAMERA_W)}; got {shape}")
    red = mx.einsum("hr,bhw,wc->brc", qr, flat, qc)
    out = mx.einsum("hr,brc,wc->bhw", qr, red, qc)
    if has_channel:
        out = mx.moveaxis(out.reshape(-1, c, CAMERA_H, CAMERA_W), 1, -1).reshape(
            *lead, CAMERA_H, CAMERA_W, c)
    else:
        out = out.reshape(*lead, CAMERA_H, CAMERA_W)
    mx.eval(out)
    return out


def projector_self_test() -> dict[str, Any]:
    """Verify ``P = Qr Qr^T (x) Qc Qc^T`` is the EXACT projector onto ``range(A^T)`` (fp64):

    (1) ``A(X - P X) == 0`` (the ker(A) residual is invisible to the scorer);
    (2) idempotence ``P(P X) == P X``;
    (3) a range(A^T) element has blind fraction 0; (4) an image on the exactly-blind rows has
    blind fraction 1.  Returns the residuals + the exact blind row/col counts.
    """
    dr, dc = _resize_matrices()
    qr, qc = _ortho_bases_f64()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((CAMERA_H, CAMERA_W))
    px = qr @ (qr.T @ x @ qc) @ qc.T
    ppx = qr @ (qr.T @ px @ qc) @ qc.T
    r_ker = float(np.abs(dr @ (x - px) @ dc.T).max())
    r_idem = float(np.abs(ppx - px).max())

    def _blind(field: np.ndarray) -> float:
        tot = float(np.sum(field * field))
        if tot == 0.0:
            return 0.0
        s = (qr.T @ field) @ qc
        return 1.0 - float(np.sum(s * s)) / tot

    y = dr.T @ rng.standard_normal((SEG_H, SEG_W)) @ dc  # in range(A^T)
    seen_blind = _blind(y)
    blind_rows = np.abs(dr).sum(axis=0) == 0.0
    xb = np.zeros_like(x)
    xb[blind_rows, :] = rng.standard_normal((int(blind_rows.sum()), CAMERA_W))
    blind_row_frac = _blind(xb)
    return {
        "max_A_of_ker_residual": r_ker,
        "idempotence_residual": r_idem,
        "seen_space_blind_fraction": seen_blind,
        "blind_row_image_blind_fraction": blind_row_frac,
        "n_exact_blind_rows": int(blind_rows.sum()),
        "n_exact_blind_cols": int((np.abs(dc).sum(axis=0) == 0.0).sum()),
        "reference_ker_residual": REFERENCE_KER_RESIDUAL,
        "matches_reference": bool(r_ker <= KER_RESIDUAL_CEILING),
    }


# ---------------------------------------------------------------------------
# The guarded, default-OFF, fail-closed consumption hook (SPEC_v10 P3 arming point).
# ---------------------------------------------------------------------------
def maybe_project_render_target(frames: Any, *,
                                enabled: bool,
                                cadence: str = "post_render",
                                backend: str = "numpy") -> Any:
    """Guarded hook: project the render target onto ``range(A)`` when ``enabled`` is True.

    DEFAULT-OFF byte-identity: when ``enabled`` is False, returns the input UNCHANGED (the
    SAME object) — a no-op the trainer can call unconditionally with zero behavior change.
    When enabled, fail-closes (raises) on an unknown cadence or a non-camera-shaped frame,
    never silently degrading.  ``backend`` selects numpy (authority) or mlx (twin).
    """
    if not enabled:
        return frames
    if cadence not in _VALID_CADENCES:
        raise RangeAProjectionError(
            f"range(A) projection cadence must be one of {_VALID_CADENCES}; got {cadence!r}")
    if backend == "mlx":
        return apply_projection_mlx(frames)
    if backend == "numpy":
        return apply_projection(frames)
    raise RangeAProjectionError(f"range(A) projection backend must be numpy|mlx; got {backend!r}")


__all__ = [
    "CAMERA_H",
    "CAMERA_W",
    "SEG_H",
    "SEG_W",
    "REFERENCE_KER_RESIDUAL",
    "RangeAProjectionError",
    "apply_projection",
    "apply_projection_mlx",
    "projector_self_test",
    "maybe_project_render_target",
]
