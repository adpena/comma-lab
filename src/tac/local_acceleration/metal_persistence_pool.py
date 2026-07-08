# SPDX-License-Identifier: MIT
"""Fused 3x3 morphological pool (min/max/mean) Metal kernel — #212 compute suite.

Why this exists
---------------
The persistence/topology (soft-clDice) loss
(``tac.boundary_math.persistence_topology_loss``) is a MEASURED hot term of the
level-set witness step: the soft-skeleton peels the class prob field with an
iterated 3x3 soft-erosion / soft-open (min/max pool) and the recall weight
smooths the GT mask with an iterated 3x3 mean pool. In pure MLX each 3x3 pool is
a **9-shift stack + reduce** (``_pool3x3_mlx``): it materializes a
``(9, M, H, W)`` intermediate (9x the memory traffic) and emits several small
Metal launches per pool. A soft-skeleton is ~17 min/max pools; the density
weight is ~4 mean pools — so a single fused kernel that computes the 3x3 window
in registers (no 9x materialization, one launch per pool) is a genuine #212
build candidate.

This module provides ONE ``mx.fast.metal_kernel`` (``persistence_pool_3x3``) that
computes the edge-clamped 3x3 min OR max OR mean over the trailing (H, W) of an
``(M, H, W)`` fp32 field in a single pass, bit-identical to the numpy authority
``tac.boundary_math.persistence_topology_loss._pool3x3_np``.

Bit-identity (max|Δ|=0, MEASURED)
---------------------------------
* ``min`` / ``max``: exact selection — order-independent, bit-exact by
  construction (no float accumulation).
* ``mean``: accumulated as a SEQUENTIAL fp32 sum of the 9 edge-clamped taps in
  the SAME ``(di, dj)`` (k=0..8) order numpy's ``np.mean(np.stack(wins), 0)``
  uses, then divided by ``9.0f``. Verified byte-for-byte equal to
  ``_pool3x3_np(x, np.mean)`` on real-shaped tensors (the 9-way reduction is
  below numpy's pairwise-sum base case, so numpy also accumulates sequentially).
  ``#pragma clang fp contract(off)`` is LOAD-BEARING here (forbids fma fusion of
  the accumulate that would break the bit-match — same rationale as the fused-R
  forward kernel).

Determinism
-----------
One thread per OUTPUT element; every thread reads its 9 taps independently and
writes one output — NO atomics, NO scatter — so the kernel is cross-process
bit-identical on GPU (unlike duplicate-index scatter; see
``tools/mlx_gpu_determinism_probe.py`` op ``persistence_pool``).

Authority note (NO-FAKE)
------------------------
This is a COMPUTE-THROUGHPUT tool on the MLX (Metal) substrate. It is **never** a
d_seg/d_pose score authority — the FP32-exact numpy reference (``_pool3x3_np``)
remains the bit-identical authority and MPS/MLX is never a score. A faster pool
is not a better score.

Dispatch / containment
----------------------
``_pool3x3_metal`` is the drop-in for ``_pool3x3_mlx``. It fires ONLY when
``persistence_pool_metal_enabled()`` is true = the env flag
``TAC_MLX_CUSTOM_PERSISTENCE_POOL`` is set AND an MLX GPU is the default device.
Otherwise the caller keeps the pure-MLX 9-shift path (CPU-correct fallback
intact). The persistence loss itself is OFF by default
(``--persistence-loss-weight 0``), so this kernel is doubly inert until both the
loss and the flag are on.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = [
    "PERSISTENCE_POOL_METAL_KERNEL_FLAG",
    "metal_persistence_pool_available",
    "persistence_pool_metal_enabled",
    "pool3x3_metal",
    "_pool3x3_metal",
]

PERSISTENCE_POOL_METAL_KERNEL_FLAG = "TAC_MLX_CUSTOM_PERSISTENCE_POOL"

_OP_CODE = {"min": 0, "max": 1, "mean": 2}

# --- Metal kernel source --------------------------------------------------
# One thread per output element (m, i, j) of an (M, H, W) field. Gather the 9
# edge-clamped taps in (di, dj) k=0..8 order (== numpy np.pad(mode='edge') then
# stack(di,dj)). op: 0=min, 1=max, 2=mean (sequential sum /9). No atomics.
_POOL_SRC = """
    #pragma clang fp contract(off)
    uint elem = thread_position_in_grid.x;
    int M = x_shape[0];
    int H = x_shape[1];
    int W = x_shape[2];
    if (elem >= (uint)(M * H * W)) return;

    int j = elem % W;
    int i = (elem / W) % H;
    int m = elem / (W * H);
    int o = op[0];

    float sum = 0.0f;
    float sel = 0.0f;
    bool first = true;
    for (int di = 0; di < 3; ++di) {
        int r = i + di - 1;
        r = r < 0 ? 0 : (r >= H ? H - 1 : r);
        for (int dj = 0; dj < 3; ++dj) {
            int c = j + dj - 1;
            c = c < 0 ? 0 : (c >= W ? W - 1 : c);
            float v = x[(m * H + r) * W + c];
            if (o == 2) {
                sum = sum + v;
            } else if (first) {
                sel = v;
                first = false;
            } else if (o == 0) {
                sel = v < sel ? v : sel;
            } else {
                sel = v > sel ? v : sel;
            }
        }
    }
    y[elem] = (o == 2) ? (sum / 9.0f) : sel;
"""

_pool_kernel = None


def _kernel():
    global _pool_kernel
    if _pool_kernel is None:
        import mlx.core as mx

        _pool_kernel = mx.fast.metal_kernel(
            name="persistence_pool_3x3",
            input_names=["x", "op"],
            output_names=["y"],
            source=_POOL_SRC,
        )
    return _pool_kernel


def metal_persistence_pool_available() -> bool:
    """True when an MLX GPU (Metal) device is the default device."""

    try:
        import mlx.core as mx

        return mx.default_device().type == mx.gpu
    except Exception:  # pragma: no cover - device introspection guard
        return False


def persistence_pool_metal_enabled() -> bool:
    """True when the env flag is set AND an MLX GPU is available.

    The flag defaults OFF (opt-in compute lever; the numpy reference stays the
    authority). Accepts ``1/true/yes/on`` (case-insensitive).
    """

    val = os.environ.get(PERSISTENCE_POOL_METAL_KERNEL_FLAG, "").strip().lower()
    if val not in ("1", "true", "yes", "on"):
        return False
    return metal_persistence_pool_available()


def pool3x3_metal(x: Any, kind: str) -> Any:
    """Edge-clamped 3x3 ``kind`` pool over the trailing (H, W) via the fused kernel.

    ``kind`` in {"min", "max", "mean"}. ``x`` is (..., H, W) fp32 (leading dims
    flattened to M internally). Bit-identical to ``_pool3x3_np(x, reduce)``.
    """

    import mlx.core as mx

    if kind not in _OP_CODE:
        raise ValueError(f"kind must be one of {sorted(_OP_CODE)}, got {kind!r}")
    if x.ndim < 2:
        raise ValueError(f"expected (..., H, W), got shape {tuple(x.shape)}")
    lead = tuple(int(d) for d in x.shape[:-2])
    h, w = int(x.shape[-2]), int(x.shape[-1])
    flat = mx.reshape(x, (-1, h, w))
    m = int(flat.shape[0])
    op = mx.array([_OP_CODE[kind]], dtype=mx.int32)
    (y,) = _kernel()(
        inputs=[flat, op],
        output_shapes=[(m, h, w)],
        output_dtypes=[flat.dtype],
        grid=(m * h * w, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return mx.reshape(y, (*lead, h, w))


def _pool3x3_metal(x: Any, kind: str) -> Any:
    """Alias mirroring ``_pool3x3_mlx``'s ``(x, kind)`` signature (dispatch drop-in)."""

    return pool3x3_metal(x, kind)
