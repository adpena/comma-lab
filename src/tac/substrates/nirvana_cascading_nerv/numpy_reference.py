# SPDX-License-Identifier: MIT
"""nirvana_cascading_nerv.numpy_reference — substrate-local numpy reference.

Per operator directive #3 2026-05-26 verbatim:
*"adversarial review against all landing recursive for math and scientific
and engineering rigor and for MLX drift minimization and portability via
numpy"*

This module is the G=NIRVANA substrate's numpy-reference surface. Per
CONSOLIDATE-OP-1 META-CONSOLIDATE-OP-2 extraction wave 2026-05-26, the 7
canonical primitives (``to_float32`` / ``linear`` / ``conv2d_nhwc`` /
``bilinear_upsample_2x_nhwc`` / ``sigmoid`` / ``sin`` / ``mean`` /
``kahan_mean``) have been EXTRACTED to the canonical sister-substrate module
``tac.local_acceleration.pr95_hnerv_numpy_reference`` so future Path 3
substrates (L/M/N/O + future) can reference (not duplicate). This module now
re-exports those 7 canonical primitives + retains the substrate-specific
composition helper ``cascade_reconstruct`` (which is specific to NIRVANA's
hierarchical residual cascade and is NOT extracted to the canonical sister
helper).

Portability per Catalog #1 device-selection-defaults discipline; enables:
(a) GHA CPU CI testing per Catalog #178 + #179 without MLX install,
(b) sister cathedral consumer cross-validation per Catalog #335,
(c) operator-portable diagnostic on non-Apple-Silicon hardware.

This module is canonical-portable: numpy only, no MLX import, no torch
import. Operable on any Python+numpy install.

Per-primitive parity bound vs MLX/PyTorch reference: ≤ 1e-5 for fp32
deterministic ops; ≤ 1e-3 for fp16 accumulation.

Back-compat: tests + downstream callers that import from this module
(e.g. ``from tac.substrates.nirvana_cascading_nerv.numpy_reference import
bilinear_upsample_2x_nhwc``) CONTINUE TO WORK unchanged — the re-exports
preserve the existing import surface. Future Path 3 substrates should
prefer the canonical sister helper module directly.
"""

from __future__ import annotations

import numpy as np

# Re-export the 7 canonical primitives (CONSOLIDATE-OP-1 META extraction
# 2026-05-26). Canonical source of truth:
# ``tac.local_acceleration.pr95_hnerv_numpy_reference``.
from tac.local_acceleration.pr95_hnerv_numpy_reference import (
    bilinear_upsample_2x_nhwc,
    conv2d_nhwc,
    kahan_mean,
    linear,
    mean,
    sigmoid,
    sin,
    to_float32,
)


# ---------------------------------------------------------------------------
# G=NIRVANA-specific composition helper (NOT extracted to canonical sister
# helper because it is substrate-specific to NIRVANA's hierarchical residual
# cascade; per CONSOLIDATE-OP-1 META extraction wave decision to scope the
# canonical sister to PR95/HNeRV-family-shared primitives only).
# ---------------------------------------------------------------------------


def cascade_reconstruct(
    base_rgb_nhwc: np.ndarray,
    residuals_nhwc: list[np.ndarray],
) -> np.ndarray:
    """Reference reconstruction for hierarchical residual cascade.

    Mirrors the canonical sequence in the MLX renderer:
        level0 → upsample → +level1_residual → upsample → +level2_residual → ...

    Args:
        base_rgb_nhwc: shape (N, H0, W0, 3) — level 0 base RGB in [0, 1]
        residuals_nhwc: list of (N, H_i, W_i, 3) residual tensors; each
            level i's H_i = 2 × H_{i-1}, W_i = 2 × W_{i-1}; residuals
            already dequantized to fp32

    Returns:
        Final RGB shape (N, H_final, W_final, 3) in [0, 1] (clamped).
    """
    current = to_float32(base_rgb_nhwc)
    for residual in residuals_nhwc:
        residual32 = to_float32(residual)
        upsampled = bilinear_upsample_2x_nhwc(current)
        if upsampled.shape != residual32.shape:
            raise ValueError(
                f"cascade_reconstruct: upsampled shape {upsampled.shape} != "
                f"residual shape {residual32.shape}"
            )
        current = upsampled + residual32
        current = np.clip(current, 0.0, 1.0)
    return current


__all__ = [
    "bilinear_upsample_2x_nhwc",
    "cascade_reconstruct",
    "conv2d_nhwc",
    "kahan_mean",
    "linear",
    "mean",
    "sigmoid",
    "sin",
    "to_float32",
]
