# SPDX-License-Identifier: MIT
"""Wyner-Ziv side-information temporal coding (frame_1 against frame_0).

Per Wyner-Ziv 1976 *The rate-distortion function for source coding with side
information at the decoder*: when the decoder has access to a correlated
sequence Y (frame_0 in our case), the rate to encode X (frame_1) is reduced
to H(X|Y), not H(X). For natural-video pairs with low motion, H(frame_1|
frame_0) is FAR smaller than H(frame_1) — typically 30-50% reduction at full
resolution and >70% reduction in the wavelet domain (residual coefficients
are sparse around zero).

v8 Path B applies this in the WAVELET DOMAIN: instead of coding frame_1's
subbands directly, we code the per-coefficient residual
``residual = frame_1_subbands - frame_0_subbands`` against the SAME Laplacian-
prior CDFs (with smaller per-class scales because the residual is sparser).

The inflate side has frame_0's decoded subbands already (the decoder works
sequentially: decode frame_0 first, then add the residual to recover frame_1).

This module is UNIQUE per UNIQUE-AND-COMPLETE-PER-METHOD. Replaces v7's
``_affine_warp_frame1_from_frame0(frame_0, pose)`` which was a STRUCTURAL
APPROXIMATION (paraxial small-angle + bilinear sample) that put approximation
error at every pixel. v8 instead codes the per-pixel residual directly.

References:
- Wyner-Ziv 1976 *The rate-distortion function for source coding with side
  information at the decoder*
- Slepian-Wolf 1973 (companion lossless theorem)
"""

from __future__ import annotations

import numpy as np


def compute_wyner_ziv_residual(
    frame0_subbands: list[np.ndarray],
    frame1_subbands: list[np.ndarray],
) -> list[np.ndarray]:
    """Compute per-subband Wyner-Ziv residual ``frame_1 - frame_0``.

    Args:
        frame0_subbands: list of subbands for frame_0 (same length as frame_1).
        frame1_subbands: list of subbands for frame_1.

    Returns:
        List of per-subband residual arrays, same shapes as inputs.

    The residual is COMPUTED on FLOAT coefficients (pre-quantization) so
    the decode side, which adds quantized frame_0 + quantized residual,
    is bit-exact under the same quantization step per subband.

    NOTE: the residual is itself FLOAT. The caller is expected to quantize
    it via the same per-subband step from wavelet_codec.PER_SUBBAND_QUANT_STEPS.
    The decode side reconstructs:
        frame_1_q = frame_0_q + residual_q  (in symbol space)
    which after dequantization gives:
        frame_1_recon = (frame_0_q + residual_q) * step
                      = frame_0_recon + residual_recon
                      = exact_frame_1 (up to per-step quantization error,
                                       same as standalone frame_1 coding)
    """
    if len(frame0_subbands) != len(frame1_subbands):
        raise ValueError(
            f"frame_0 has {len(frame0_subbands)} subbands; "
            f"frame_1 has {len(frame1_subbands)} subbands"
        )
    residuals: list[np.ndarray] = []
    for s, (f0, f1) in enumerate(zip(frame0_subbands, frame1_subbands)):
        if f0.shape != f1.shape:
            raise ValueError(
                f"subband {s}: frame_0 shape {f0.shape} != frame_1 shape {f1.shape}"
            )
        residuals.append((f1.astype(np.float64) - f0.astype(np.float64)))
    return residuals


def reconstruct_frame1_from_frame0_and_residual(
    frame0_subbands_q: list[np.ndarray],
    residual_subbands_q: list[np.ndarray],
) -> list[np.ndarray]:
    """Inverse Wyner-Ziv: add quantized residual to quantized frame_0 subbands.

    Args:
        frame0_subbands_q: list of int8 quantized frame_0 subbands.
        residual_subbands_q: list of int8 quantized residual subbands.

    Returns:
        List of int8 reconstructed frame_1 subbands (in symbol space; caller
        must dequantize via per-subband step).

    Symbol-space addition can overflow int8 range; we clamp to the same
    [-QUANT_CLAMP_T, +QUANT_CLAMP_T] range that wavelet_codec.quantize_subband
    uses (re-imported here to keep this module's invariants explicit).
    """
    from .wavelet_codec import QUANT_CLAMP_T

    if len(frame0_subbands_q) != len(residual_subbands_q):
        raise ValueError(
            f"frame_0 {len(frame0_subbands_q)} subbands; "
            f"residual {len(residual_subbands_q)} subbands"
        )
    out: list[np.ndarray] = []
    for s, (f0_q, r_q) in enumerate(zip(frame0_subbands_q, residual_subbands_q)):
        if f0_q.shape != r_q.shape:
            raise ValueError(
                f"subband {s}: frame_0 shape {f0_q.shape} != "
                f"residual shape {r_q.shape}"
            )
        # Sum in wider int to avoid wrap; clamp to QUANT_CLAMP_T range
        summed = f0_q.astype(np.int32) + r_q.astype(np.int32)
        summed = np.clip(summed, -QUANT_CLAMP_T, QUANT_CLAMP_T)
        out.append(summed.astype(np.int8))
    return out
