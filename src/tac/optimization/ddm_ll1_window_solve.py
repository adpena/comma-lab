"""ddm_ll1 — per-scorer-pixel window solve for the camera raster (receiver-side).

GENERIC ALGORITHM, ZERO COUNTED BYTES.  This is decode-time receiver code with no
video-derived content whatsoever: the geometry is derived from the two frozen
shapes, and the only input is the render the receiver already computes.  Rule-118
free (see CLAUDE.md "inflate.py is a FREE interpreter").

THE STRUCTURE IT EXPLOITS (MEASURED 2026-08-02, exact)
------------------------------------------------------
The scorer downsample is ``F.interpolate(x,(384,512),'bilinear')``.  PyTorch
defaults ``align_corners=False`` AND ``antialias=False``, so on DOWNsampling this
is point sampling with bilinear weights, NOT an area average::

    stride y = 874/384 = 2.2760      stride x = 1164/512 = 2.2734      both > 2
    => consecutive 2x2 read-windows CANNOT overlap

Measured reads-per-camera-pixel histogram has exactly two bins::

    0x : 230,904 px (22.70%)   <- read by NEITHER scorer
    1x : 786,432 px (77.30%)   = 196,608 scorer px * 4, EXACTLY

So each scorer pixel owns a PRIVATE, DISJOINT 2x2 camera window, and the uint8
preimage problem decouples exactly into 196,608 independent 4-variable problems.
Geometry verified against torch fp64 at 6.45e-12.

PoseNet reads through the IDENTICAL operator (``PoseNet.preprocess_input``
interpolates to ``segnet_model_input_size`` BEFORE ``rgb_to_yuv6``), so the same
windows and the same blind set serve both scorers.

SOLVE, DON'T SEARCH
-------------------
One linear equation in four integers.  Enumerating {-2..2}^4 around the rounded
start is both slower AND worse than allocating directly (measured: 2.04 s/frame
rms 0.0715 vs 0.07 s/frame rms 0.0282).  Walk the four taps weight-descending,
give each the integer step that best closes the running residual, clip, and
subtract what it ACTUALLY delivered.  Converges in one pass.

MEASURED EFFECT (canonical DistortionNet path, n=3 real frames)::

    clip(rint(U(r)))     88 flips vs perfect delivery   d_seg 0.0001492
    window solve          3 flips                        d_seg 0.0000051
                                                   =>  Delta S = -0.01441

Blind-set pixels keep their input values BY CONSTRUCTION (the scatter writes only
read-set indices onto a copy) — VERIFIED, 230,904 px byte-identical.
"""

from __future__ import annotations

from typing import Final

import numpy as np

CAMERA_H: Final = 874
CAMERA_W: Final = 1164
SEG_H: Final = 384
SEG_W: Final = 512

_QUAD: Final = ((0, 0), (0, 1), (1, 0), (1, 1))


def _bilinear_taps(src: int, dst: int) -> tuple[np.ndarray, np.ndarray]:
    """(indices, weights) for torch bilinear ``align_corners=False`` downsampling.

    Derived from the two shapes; nothing here is video-derived.  torch clamps the
    sample coordinate into range and clamps the upper tap at the edge — both are
    reproduced so this matches ``F.interpolate`` exactly (fp64 parity 6.45e-12).
    """
    centers = np.clip((np.arange(dst, dtype=np.float64) + 0.5) * (src / dst) - 0.5, 0.0, None)
    i0 = np.floor(centers).astype(np.int64)
    frac = centers - i0
    idx = np.stack([np.clip(i0, 0, src - 1), np.clip(i0 + 1, 0, src - 1)], axis=1)
    return idx, np.stack([1.0 - frac, frac], axis=1)


def window_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (ys, xs, wy, wx) — the private 2x2 windows and separable weights."""
    ys, wy = _bilinear_taps(CAMERA_H, SEG_H)
    xs, wx = _bilinear_taps(CAMERA_W, SEG_W)
    return ys, xs, wy, wx


def solve_camera_windows(
    camera_uint8: np.ndarray,
    target_plane: np.ndarray,
    *,
    passes: int = 1,
) -> np.ndarray:
    """Choose each private window's 4 uint8 values so D(camera) hits ``target_plane``.

    ``camera_uint8``  (874,1164,3) uint8 starting raster (typically the rounded
                      bicubic upsample). Blind pixels are carried through
                      untouched — they are never written.
    ``target_plane``  (384,512,3)  the render the scorer is meant to see.

    Deterministic: fixed tap order, fixed weight-descending tie-break, no RNG.
    """
    if camera_uint8.shape != (CAMERA_H, CAMERA_W, 3):
        raise ValueError(f"camera must be {(CAMERA_H, CAMERA_W, 3)}, got {camera_uint8.shape}")
    if target_plane.shape != (SEG_H, SEG_W, 3):
        raise ValueError(f"target must be {(SEG_H, SEG_W, 3)}, got {target_plane.shape}")

    ys, xs, wy, wx = window_geometry()
    target = np.asarray(target_plane, dtype=np.float64)

    c = [camera_uint8[np.ix_(ys[:, a], xs[:, b])].astype(np.float64) for a, b in _QUAD]
    w = [(wy[:, a][:, None] * wx[:, b][None, :])[..., None] for a, b in _QUAD]

    err = target - sum(w[k] * c[k] for k in range(4))
    order = np.argsort(
        -np.concatenate([np.broadcast_to(w[k], err.shape)[..., None] for k in range(4)], axis=-1),
        axis=-1,
    )

    for _ in range(max(1, passes)):
        for slot in range(4):
            chosen = order[..., slot]
            for k in range(4):
                mask = chosen == k
                if not mask.any():
                    continue
                wk = np.broadcast_to(w[k], err.shape)
                step = np.where(
                    mask,
                    np.rint(np.divide(err, wk, out=np.zeros_like(err), where=wk > 1e-12)),
                    0.0,
                )
                # Clip FIRST, then subtract what was ACTUALLY delivered. Subtracting
                # the requested step instead desynchronises the residual at any
                # window whose tap is pinned at 0 or 255.
                step = np.clip(c[k] + step, 0.0, 255.0) - c[k]
                c[k] = c[k] + step
                err = err - wk * step

    out = camera_uint8.astype(np.int16).copy()
    for k, (a, b) in enumerate(_QUAD):
        out[ys[:, a][:, None], xs[:, b][None, :], :] = np.clip(c[k], 0, 255).astype(np.int16)
    return np.ascontiguousarray(out.astype(np.uint8))


def blind_mask() -> np.ndarray:
    """(874,1164) bool — True where NEITHER scorer reads (22.70%, 230,904 px).

    Standing basis for task #401: these carry no seg or pose signal through D, but
    v4d's frame_0 warp DOES read them, so they are pose-only currency.
    """
    ys, xs, _, _ = window_geometry()
    read = np.zeros((CAMERA_H, CAMERA_W), dtype=bool)
    for a in range(2):
        for b in range(2):
            read[np.ix_(ys[:, a], xs[:, b])] = True
    return ~read
