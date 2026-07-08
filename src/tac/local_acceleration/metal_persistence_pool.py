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
    "PERSISTENCE_POOL_CERT_FP_ENV",
    "metal_persistence_pool_available",
    "persistence_pool_metal_enabled",
    "persistence_pool_certified_fingerprint",
    "persistence_pool_fingerprint_ok",
    "emit_persistence_pool_path_marker",
    "reset_persistence_pool_path_markers",
    "pool3x3_metal",
    "_pool3x3_metal",
]

PERSISTENCE_POOL_METAL_KERNEL_FLAG = "TAC_MLX_CUSTOM_PERSISTENCE_POOL"

# Optional certified {chip, macos_build, mlx_version} fingerprint the bit-identity of the fused pool
# kernel was MEASURED on (JSON dict). When set + it MISMATCHES the host, the dispatch fails CLOSED to
# pure-MLX LOUDLY (confound F2 fix, mirrors the safe-compile per-chip trust). Absent => permissive
# (min/max are order-independent bit-exact by construction; mean is verified — the same
# empty-fingerprint = legacy/back-compat semantics as tac.mlx_safe_compile.manifest_fingerprint_ok).
PERSISTENCE_POOL_CERT_FP_ENV = "TAC_MLX_CUSTOM_PERSISTENCE_POOL_CERT_FP"

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

    NOTE (confound F2): this is the FLAG+DEVICE gate only. The fingerprint gate
    (:func:`persistence_pool_fingerprint_ok`) is a SEPARATE fail-closed check the
    dispatch site (``_smooth_density_mlx``) ANDs in, so this predicate keeps its
    historical flag+GPU meaning (existing tests) while a stale-cert host still
    falls back LOUDLY, not silently.
    """

    val = os.environ.get(PERSISTENCE_POOL_METAL_KERNEL_FLAG, "").strip().lower()
    if val not in ("1", "true", "yes", "on"):
        return False
    return metal_persistence_pool_available()


def persistence_pool_certified_fingerprint() -> "dict[str, str] | None":
    """The certified {chip, macos_build, mlx_version} fingerprint recorded in
    :data:`PERSISTENCE_POOL_CERT_FP_ENV` (JSON dict), or ``None`` when unset/unparseable.

    ``None`` => no recorded cert => the fingerprint gate is permissive (legacy/back-compat;
    min/max are bit-exact by construction, mean verified) — the SAME empty-fingerprint semantics
    as ``tac.mlx_safe_compile.manifest_fingerprint_ok``."""
    raw = os.environ.get(PERSISTENCE_POOL_CERT_FP_ENV, "").strip()
    if not raw:
        return None
    try:
        import json

        fp = json.loads(raw)
    except Exception:
        return None
    if not isinstance(fp, dict):
        return None
    return {str(k): str(v) for k, v in fp.items()}


# Hot-path cache for the DEFAULT (no explicit recorded/host) fingerprint verdict:
# ``host_fingerprint()`` spawns sysctl/sw_vers subprocesses, and ``_smooth_density_mlx`` runs EVERY
# training step — re-deriving per call would be a per-step performance regression (own round-1
# review catch). The verdict is stable within a process for a given cert-env value, so cache keyed
# by the raw env string (tests that change/clear the env get a distinct key => fresh verdict).
_FP_OK_CACHE: "dict[str, tuple[bool, str]]" = {}


def persistence_pool_fingerprint_ok(
    recorded: "dict[str, str] | None" = None,
    host: "dict[str, str] | None" = None,
) -> "tuple[bool, str]":
    """``(ok, reason)`` for the pool kernel's certified fingerprint vs the host — the confound-F2
    fail-closed gate, consistent with the safe-compile sibling.

    Reuses ``tac.mlx_safe_compile.fingerprint_matches`` / ``host_fingerprint`` when importable. A
    missing recorded fingerprint (default) is OK (permissive: min/max bit-exact by construction, mean
    verified — matching the sibling's empty-fingerprint = legacy back-compat rule). A recorded cert
    that MISMATCHES the host is fail-closed (``False``) so the dispatch site can fall back LOUDLY.
    Best-effort: if the helpers are unavailable it stays permissive (cannot gate what it cannot read).
    The DEFAULT path (no explicit args — the per-step dispatch call) is CACHED per cert-env value."""
    if recorded is None and host is None:
        key = os.environ.get(PERSISTENCE_POOL_CERT_FP_ENV, "").strip()
        hit = _FP_OK_CACHE.get(key)
        if hit is not None:
            return hit
        verdict = _fingerprint_ok_uncached(None, None)
        _FP_OK_CACHE[key] = verdict
        return verdict
    return _fingerprint_ok_uncached(recorded, host)


def _fingerprint_ok_uncached(
    recorded: "dict[str, str] | None", host: "dict[str, str] | None"
) -> "tuple[bool, str]":
    fp = recorded if recorded is not None else persistence_pool_certified_fingerprint()
    if not fp:
        return True, ("no certified fingerprint recorded (legacy/unscoped; min/max bit-exact by "
                      "construction, mean verified) -> permissive")
    try:
        from tac.mlx_safe_compile import fingerprint_matches, host_fingerprint
    except Exception:  # pragma: no cover - safe-compile helpers absent
        return True, "safe-compile fingerprint helpers unavailable -> cannot gate (permissive)"
    h = host if host is not None else host_fingerprint()
    if fingerprint_matches(fp, h):
        return True, "persistence-pool kernel cert fingerprint matches host"
    return False, (
        f"STALE persistence-pool kernel cert: {fp} != host {h} — recertify bit-identity on this host "
        "(Metal codegen is per-chip; fail-closed to pure-MLX)"
    )


# One-time compute-path markers, keyed by (path, reason) so the LOUD row prints ONCE per distinct
# resolution per process (not every training step). Reset helper is for tests.
_POOL_PATH_MARKER_EMITTED: "set[str]" = set()


def reset_persistence_pool_path_markers() -> None:
    """Clear the one-time marker set (test hook)."""
    _POOL_PATH_MARKER_EMITTED.clear()


def emit_persistence_pool_path_marker(path: str, reason: str, *, emit=None) -> bool:
    """LOUD ONE-TIME marker when the persistence-pool compute path resolves (confound F2: the
    kernel<->pure-MLX flip is never silent + gives cross-run compute-path provenance).

    ``path`` is ``"metal_kernel"`` (the fused kernel fired) or ``"pure_mlx_fallback"`` (the 9-shift
    pure-MLX path ran). A flip AWAY from the kernel while it was expected (flag+GPU on) sets
    ``confound_alarm=True``. Returns True iff it emitted (first time for this ``(path, reason)`` this
    process). ``emit`` defaults to a flushed ``print`` (the run log captures the JSON row)."""
    key = f"{path}:{reason}"
    if key in _POOL_PATH_MARKER_EMITTED:
        return False
    _POOL_PATH_MARKER_EMITTED.add(key)
    import json

    row = {
        "stage": "persistence_pool_compute_path",
        "confound_alarm": bool(path != "metal_kernel"),
        "compute_path": str(path),
        "reason": str(reason),
        "flag": PERSISTENCE_POOL_METAL_KERNEL_FLAG,
        "axis": "[macOS-MLX compute] NON-PROMOTABLE (score-neutral: forward-only, stop_gradient'd)",
    }
    _emit = emit if emit is not None else _default_marker_emit
    _emit(json.dumps(row))
    return True


def _default_marker_emit(line: str) -> None:
    print(line, flush=True)


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
