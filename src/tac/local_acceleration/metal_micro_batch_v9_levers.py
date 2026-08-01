"""Fused map kernels for the V9 micro-batch-only segmentation levers.

The reductions deliberately live in ``levelset_micro_batch_loss``: each pair owns
its denominator and only the resulting scalars are averaged.  This module fuses
the pixel map construction, which is the portion that is safe to share across the
batch.  Every Metal forward has an explicit fused Metal transpose/VJP for its
theta-bearing inputs; the pure-MLX implementation remains the CPU/non-fp32 fallback
and functional authority.  Gradients of static provider fields use safe generic MLX
expressions where a custom reduction would add dispatch risk without training value.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "chroma_squared_map",
    "chroma_squared_map_reference",
    "metal_micro_batch_v9_available",
    "phase_squared_map",
    "phase_squared_map_reference",
    "temporal_squared_map",
    "temporal_squared_map_reference",
    "v9_lever_backend_details",
    "v9_lever_backend_receipt",
    "verify_v9_lever_backend_execution",
]

_KINDS = ("chroma", "phase", "temporal")


def _new_backend_state() -> dict[str, Any]:
    return {
        "selected_backend": "unused",
        "forward_status": "unused",
        "backward_status": "unused",
        "verified_backend": None,
        "error": None,
        "generation": 0,
        # These arrays are intentionally private.  Verification evaluates the arrays emitted by
        # this module itself, so a caller cannot bless an unrelated successful MLX expression.
        "_pending_forward": (),
        "_pending_backward": (),
    }


_BACKEND_STATE = {kind: _new_backend_state() for kind in _KINDS}
_DEVICE_READY: bool | None = None


def v9_lever_backend_receipt() -> dict[str, str]:
    """Verified backend for the latest call to each lever.

    ``"metal"`` is returned only after :func:`verify_v9_lever_backend_execution` has
    successfully evaluated both the fused forward and its fused transpose/VJP.  Merely
    constructing MLX's lazy graph returns ``"metal_planned"``; a failed evaluation returns
    ``"metal_failed"``; an explicitly forward-only check is ``"metal_forward_verified"``.
    This intentionally makes old probes fail closed until they perform the explicit
    execution-verification step.
    """

    out: dict[str, str] = {}
    for kind, state in _BACKEND_STATE.items():
        verified = state["verified_backend"]
        selected = str(state["selected_backend"])
        if verified == "metal":
            out[kind] = "metal"
        elif verified == "metal_forward_only":
            out[kind] = "metal_forward_verified"
        elif selected == "metal" and (
            state["forward_status"] == "failed" or state["backward_status"] == "failed"
        ):
            out[kind] = "metal_failed"
        elif selected == "metal":
            out[kind] = "metal_planned"
        else:
            out[kind] = selected
    return out


def v9_lever_backend_details() -> dict[str, dict[str, Any]]:
    """Non-array backend state for diagnostics and fail-closed admission receipts."""

    return {
        kind: {key: value for key, value in state.items() if not key.startswith("_pending_")}
        for kind, state in _BACKEND_STATE.items()
    }


def _select_backend(kind: str, selected_backend: str) -> int:
    state = _BACKEND_STATE[kind]
    state["generation"] = int(state["generation"]) + 1
    state["selected_backend"] = str(selected_backend)
    state["forward_status"] = "selected" if selected_backend == "metal" else "reference"
    state["backward_status"] = "not_built" if selected_backend == "metal" else "reference"
    state["verified_backend"] = None
    state["error"] = None
    state["_pending_forward"] = ()
    state["_pending_backward"] = ()
    return int(state["generation"])


def _record_pending(kind: str, leg: str, arrays: Any) -> None:
    if _BACKEND_STATE[kind]["selected_backend"] != "metal":
        return
    packed = tuple(arrays) if isinstance(arrays, (tuple, list)) else (arrays,)
    _BACKEND_STATE[kind][f"_pending_{leg}"] = packed
    _BACKEND_STATE[kind][f"{leg}_status"] = "planned"


def verify_v9_lever_backend_execution(kind: str, *, require_backward: bool = True) -> dict[str, Any]:
    """Evaluate this module's pending fused arrays and return the verified backend state.

    MLX arrays are lazy.  A kernel call proves selection only; it is not evidence that Metal
    compiled or executed.  This function owns the synchronization boundary used by faithful
    probes: it evaluates the exact forward arrays recorded by the selected custom function and,
    by default, every theta-bearing gradient array emitted by its fused custom VJP.  Static
    provider-map gradients are diagnostic only and do not authorize training.  State changes to
    ``evaluated`` and ``verified_backend="metal"`` only after that call returns successfully.
    Headless Metal, compile errors, or runtime errors are recorded as failures and re-raised.
    """

    if kind not in _BACKEND_STATE:
        raise ValueError(f"unknown V9 lever backend {kind!r}")
    state = _BACKEND_STATE[kind]
    if state["selected_backend"] != "metal":
        raise RuntimeError(
            f"V9 {kind} Metal verification REFUSE: selected backend is "
            f"{state['selected_backend']!r}"
        )
    forward = tuple(state["_pending_forward"])
    backward = tuple(state["_pending_backward"])
    if not forward:
        raise RuntimeError(f"V9 {kind} Metal verification REFUSE: no pending fused forward")
    if require_backward and not backward:
        raise RuntimeError(f"V9 {kind} Metal verification REFUSE: no pending fused backward/VJP")
    import mlx.core as mx

    try:
        mx.eval(*forward)
    except Exception as exc:
        state["forward_status"] = "failed"
        state["verified_backend"] = None
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["_pending_forward"] = ()
        state["_pending_backward"] = ()
        raise
    state["forward_status"] = "evaluated"
    if require_backward:
        try:
            mx.eval(*backward)
        except Exception as exc:
            state["backward_status"] = "failed"
            state["verified_backend"] = None
            state["error"] = f"{type(exc).__name__}: {exc}"
            state["_pending_forward"] = ()
            state["_pending_backward"] = ()
            raise
        state["backward_status"] = "evaluated"
        state["verified_backend"] = "metal"
    else:
        state["backward_status"] = "not_verified"
        state["verified_backend"] = "metal_forward_only"
    state["error"] = None
    state["_pending_forward"] = ()
    state["_pending_backward"] = ()
    return v9_lever_backend_details()[kind]


def metal_micro_batch_v9_available() -> bool:
    """Whether Metal is selected *and actually initializes* (lazy-load failures fall soft)."""
    global _DEVICE_READY
    try:
        import mlx.core as mx

        if mx.default_device().type != mx.gpu:
            return False
        if _DEVICE_READY is not None:
            return _DEVICE_READY
        # ``default_device`` can report GPU in a headless sandbox while the first real command
        # raises ``metal::load_device``. Force that lazy initialization once before dispatch.
        probe = mx.zeros((1,), dtype=mx.float32)
        mx.eval(probe)
        _DEVICE_READY = True
        return True
    except Exception:
        _DEVICE_READY = False
        return False


def chroma_squared_map_reference(frame_rgb: Any, target_chroma: Any) -> Any:
    """BT.601 chroma squared error, summed over RGB, shape ``(B,H,W)``."""
    import mlx.core as mx

    luma = (
        0.299 * frame_rgb[..., 0:1]
        + 0.587 * frame_rgb[..., 1:2]
        + 0.114 * frame_rgb[..., 2:3]
    )
    chroma = frame_rgb - luma
    return mx.sum(mx.square(chroma - target_chroma), axis=-1)


def phase_squared_map_reference(
    signed: Any, direction: Any, reference: Any, eps: float = 1e-6
) -> Any:
    """Witness tie-coordinate squared residual, vectorized over ``B``."""
    import mlx.core as mx

    margin = mx.maximum(signed, 0.0)
    right = mx.pad(margin[:, :, 1:], [(0, 0), (0, 0), (0, 1)])
    down = mx.pad(margin[:, 1:, :], [(0, 0), (0, 1), (0, 0)])
    partner = mx.where(direction < 0.5, right, down)
    tie = margin / (margin + partner + float(eps))
    return mx.square(tie - reference)


def temporal_squared_map_reference(g1: Any, g0_warped: Any, class_mask: Any) -> Any:
    """Class-selected temporal probability residual, shape ``(B,H,W)``."""
    import mlx.core as mx

    return mx.sum(mx.square(g1 - g0_warped) * class_mask, axis=-1)


_CHROMA_SRC = r"""
    uint i = thread_position_in_grid.x;
    if (i >= (uint)(frame_shape[0] * frame_shape[1] * frame_shape[2])) return;
    uint p = i * 3;
    float r = frame[p];
    float g = frame[p + 1];
    float b = frame[p + 2];
    float y = 0.299f * r + 0.587f * g + 0.114f * b;
    float dr = (r - y) - target[p];
    float dg = (g - y) - target[p + 1];
    float db = (b - y) - target[p + 2];
    out[i] = dr * dr + dg * dg + db * db;
"""

# NAMING CONSTRAINT (do NOT rename ``sdf`` back to ``signed``): ``mx.fast.metal_kernel`` splices each
# ``input_names`` entry VERBATIM into the generated Metal signature (``const device float* <name>``)
# and derives ``<name>_shape`` from it.  ``signed`` is a Metal/C++ type keyword, so an input named
# ``signed`` makes the kernel fail to COMPILE ("invalid parameter name: 'signed' is a keyword") on
# every dispatch.  The Python-facing argument stays ``signed`` (public API); only the buffer identifier
# crossing into Metal is ``sdf``.  MEASURED 2026-08-01: before this rename the phase forward/VJP
# kernels had never once compiled -- not since they were written on 2026-07-12 (d880211c96, whose
# own message records "Metal-verify ... owed" because the authoring sandbox had no Metal). CI
# cannot run MLX either, so nothing executed them for 20 days.
_PHASE_SRC = r"""
    uint i = thread_position_in_grid.x;
    int B = sdf_shape[0];
    int H = sdf_shape[1];
    int W = sdf_shape[2];
    if (i >= (uint)(B * H * W)) return;
    int col = i % W;
    int row = (i / W) % H;
    float m = sdf[i] > 0.0f ? sdf[i] : 0.0f;
    bool right = direction[i] < 0.5f;
    int q = (int)i;
    bool has_q = true;
    if (right) {
        if (col + 1 < W) q += 1;
        else has_q = false;
    } else {
        if (row + 1 < H) q += W;
        else has_q = false;
    }
    float mq = has_q && sdf[q] > 0.0f ? sdf[q] : 0.0f;
    float t = m / (m + mq + eps[0]);
    float d = t - reference[i];
    out[i] = d * d;
"""

_TEMPORAL_SRC = r"""
    uint i = thread_position_in_grid.x;
    if (i >= (uint)(g1_shape[0] * g1_shape[1] * g1_shape[2])) return;
    uint p = i * 3;
    float d0 = g1[p] - g0w[p];
    float d1 = g1[p + 1] - g0w[p + 1];
    float d2 = g1[p + 2] - g0w[p + 2];
    out[i] = d0 * d0 * class_mask[0]
           + d1 * d1 * class_mask[1]
           + d2 * d2 * class_mask[2];
"""

# One thread per pixel writes the theta-bearing RGB gradient.  The target chroma is a static
# provider; its diagnostic VJP is a dead-in-training generic MLX expression in the custom VJP.
_CHROMA_VJP_SRC = r"""
    uint i = thread_position_in_grid.x;
    if (i >= (uint)(frame_shape[0] * frame_shape[1] * frame_shape[2])) return;
    uint p = i * 3;
    float r = frame[p];
    float g = frame[p + 1];
    float b = frame[p + 2];
    float y = 0.299f * r + 0.587f * g + 0.114f * b;
    float d0 = (r - y) - target[p];
    float d1 = (g - y) - target[p + 1];
    float d2 = (b - y) - target[p + 2];
    float two_cot = 2.0f * cot[i];
    float dsum = d0 + d1 + d2;
    g_frame[p]     = two_cot * (d0 - 0.299f * dsum);
    g_frame[p + 1] = two_cot * (d1 - 0.587f * dsum);
    g_frame[p + 2] = two_cot * (d2 - 0.114f * dsum);
"""

# The phase tie at pixel p depends on signed[p] and at most one right/down partner.  Therefore
# dL/dsigned[i] is the sum of its own term plus at most two predecessor terms (left and up).
# Computing one output element per thread avoids nondeterministic atomics while retaining O(BHW).
#
# ACCURACY DIRECTION (MEASURED 2026-08-01, ddm_mk1 -- do NOT "fix" this kernel toward the MLX
# reference).  This kernel is the MORE ACCURATE side of the pair.  Against an independent float64
# evaluation at (B,H,W)=(2,384,512): this kernel is 1.60 ulp-of-field from truth, the MLX
# reference (``phase_squared_map_reference`` differentiated by MLX) is 25.6 ulp -- ~16x worse.
# Cause: MLX forms dL/dm for ``tie = m/den`` as ``g/den - g*tie/den``, a difference of two
# O(g/den) terms that cancels catastrophically as ``tie -> 1``; the closed form below,
# ``(mq + eps)/den**2``, is algebraically identical (float64 agreement 2.9e-15, both matching a
# central finite difference to 3.3e-9) but has no subtraction and therefore no cancellation.
# Ruled out as causes, both MEASURED: the MLX GPU stream (reference GPU vs reference CPU is
# bit-identical, 0/393216 elements differing) and any matmul precision floor (there is no matmul
# here).  The parity test therefore bounds the disagreement by the REFERENCE's cancellation floor
# -- see ``_assert_phase_signed_grad_parity`` in test_levelset_micro_batch_loss.py.
_PHASE_VJP_SRC = r"""
    uint i = thread_position_in_grid.x;
    int B = sdf_shape[0];
    int H = sdf_shape[1];
    int W = sdf_shape[2];
    if (i >= (uint)(B * H * W)) return;
    int col = i % W;
    int row = (i / W) % H;
    float si = sdf[i];
    float m = si > 0.0f ? si : 0.0f;
    float grad = 0.0f;

    // Pixel i's own tie coordinate.
    bool own_right = direction[i] < 0.5f;
    int q = (int)i;
    bool has_q = true;
    if (own_right) {
        if (col + 1 < W) q += 1;
        else has_q = false;
    } else {
        if (row + 1 < H) q += W;
        else has_q = false;
    }
    float mq = has_q && sdf[q] > 0.0f ? sdf[q] : 0.0f;
    float den = m + mq + eps[0];
    float tie = m / den;
    float common = 2.0f * (tie - reference[i]) * cot[i];
    if (si > 0.0f) grad += common * (mq + eps[0]) / (den * den);
    if (si > 0.0f && col > 0) {
        int p = (int)i - 1;
        // The left predecessor chooses its right neighbour only when direction[p] < 0.5.
        if (direction[p] < 0.5f) {
            float pm = sdf[p] > 0.0f ? sdf[p] : 0.0f;
            float pden = pm + m + eps[0];
            float ptie = pm / pden;
            float pcommon = 2.0f * (ptie - reference[p]) * cot[p];
            grad += pcommon * (-pm) / (pden * pden);
        }
    }
    if (si > 0.0f && row > 0) {
        int p = (int)i - W;
        // The upper predecessor chooses its down neighbour for every non-right selector,
        // including NaN, matching mx.where(direction < 0.5, right, down).
        if (!(direction[p] < 0.5f)) {
            float pm = sdf[p] > 0.0f ? sdf[p] : 0.0f;
            float pden = pm + m + eps[0];
            float ptie = pm / pden;
            float pcommon = 2.0f * (ptie - reference[p]) * cot[p];
            grad += pcommon * (-pm) / (pden * pden);
        }
    }
    g_sdf[i] = grad;
"""

# One thread per pixel writes the two theta-bearing g1/g0 gradients.  The canonical class mask is a
# static provider, so its diagnostic gradient is the safe generic MLX batch reduction in the VJP.
_TEMPORAL_VJP_SRC = r"""
    uint i = thread_position_in_grid.x;
    uint pixels = (uint)(g1_shape[0] * g1_shape[1] * g1_shape[2]);
    if (i >= pixels) return;
    uint p = i * 3;
    float two_cot = 2.0f * cot[i];
    for (uint c = 0; c < 3; ++c) {
        float d = g1[p + c] - g0w[p + c];
        float grad = two_cot * d * class_mask[c];
        g_g1[p + c] = grad;
        g_g0w[p + c] = -grad;
    }
"""

_KERNEL_CACHE: dict[str, Any] = {}
_CHROMA_FN = None
_TEMPORAL_FN = None
_PHASE_FNS: dict[float, Any] = {}


def _kernel(kind: str):
    import mlx.core as mx

    hit = _KERNEL_CACHE.get(kind)
    if hit is not None:
        return hit
    if kind == "chroma":
        hit = mx.fast.metal_kernel(
            name="micro_batch_v9_chroma_map",
            input_names=["frame", "target"], output_names=["out"], source=_CHROMA_SRC,
        )
    elif kind == "phase":
        hit = mx.fast.metal_kernel(
            name="micro_batch_v9_phase_map",
            input_names=["sdf", "direction", "reference", "eps"],
            output_names=["out"], source=_PHASE_SRC,
        )
    elif kind == "temporal":
        hit = mx.fast.metal_kernel(
            name="micro_batch_v9_temporal_map",
            input_names=["g1", "g0w", "class_mask"], output_names=["out"],
            source=_TEMPORAL_SRC,
        )
    elif kind == "chroma_vjp":
        hit = mx.fast.metal_kernel(
            name="micro_batch_v9_chroma_map_vjp",
            input_names=["frame", "target", "cot"],
            output_names=["g_frame"], source=_CHROMA_VJP_SRC,
        )
    elif kind == "phase_vjp":
        hit = mx.fast.metal_kernel(
            name="micro_batch_v9_phase_map_vjp",
            input_names=["sdf", "direction", "reference", "eps", "cot"],
            output_names=["g_sdf"],
            source=_PHASE_VJP_SRC,
        )
    elif kind == "temporal_vjp":
        hit = mx.fast.metal_kernel(
            name="micro_batch_v9_temporal_map_vjp",
            input_names=["g1", "g0w", "class_mask", "cot"],
            output_names=["g_g1", "g_g0w"],
            source=_TEMPORAL_VJP_SRC,
        )
    else:  # pragma: no cover - internal programming error
        raise ValueError(f"unknown V9 lever kernel {kind!r}")
    _KERNEL_CACHE[kind] = hit
    return hit


def _raw_primal(value):
    return value[0] if isinstance(value, (tuple, list)) and len(value) == 1 else value


def _metal_eligible(*arrays: Any) -> bool:
    import mlx.core as mx

    if not metal_micro_batch_v9_available():
        return False
    # Kernels accumulate in float. Keep other dtypes on the differentiable reference.
    return all(getattr(a, "dtype", None) == mx.float32 for a in arrays)


def _require_equal_shape(name: str, lhs: Any, rhs: Any) -> None:
    if tuple(lhs.shape) != tuple(rhs.shape):
        raise ValueError(
            f"{name} inputs must have identical shapes, got {tuple(lhs.shape)} and {tuple(rhs.shape)}"
        )


def _require_rgb_batch(name: str, value: Any) -> None:
    shape = tuple(value.shape)
    if len(shape) != 4 or int(shape[-1]) != 3 or any(int(d) <= 0 for d in shape[:-1]):
        raise ValueError(f"{name} must have shape (B,H,W,3), got {shape}")


def _require_map_batch(name: str, value: Any) -> None:
    shape = tuple(value.shape)
    if len(shape) != 3 or any(int(d) <= 0 for d in shape):
        raise ValueError(f"{name} must have shape (B,H,W), got {shape}")


def _make_chroma_metal():
    import mlx.core as mx

    @mx.custom_function
    def fn(frame, target):
        shape = tuple(int(x) for x in frame.shape[:-1])
        (out,) = _kernel("chroma")(
            inputs=[frame, target], output_shapes=[shape], output_dtypes=[frame.dtype],
            grid=(int(frame.size // 3), 1, 1), threadgroup=(256, 1, 1),
        )
        _record_pending("chroma", "forward", out)
        return out

    @fn.vjp
    def _vjp(primals, cotangent, output):
        frame, target = primals
        cot = _raw_primal(cotangent)
        (g_frame,) = _kernel("chroma_vjp")(
            inputs=[frame, target, cot],
            output_shapes=[frame.shape], output_dtypes=[frame.dtype],
            grid=(int(frame.size // 3), 1, 1), threadgroup=(256, 1, 1),
        )
        # Static provider VJP (dead under model/theta differentiation, complete for diagnostics).
        luma = (
            0.299 * frame[..., 0:1]
            + 0.587 * frame[..., 1:2]
            + 0.114 * frame[..., 2:3]
        )
        delta = frame - luma - target
        g_target = -2.0 * cot[..., None] * delta
        grads = (g_frame, g_target)
        _record_pending("chroma", "backward", g_frame)
        return tuple(grads)

    return fn


def _make_phase_metal(eps: float):
    import mlx.core as mx

    eps_value = float(eps)

    @mx.custom_function
    def fn(signed, direction, reference):
        eps_a = mx.array([eps_value], dtype=mx.float32)
        (out,) = _kernel("phase")(
            inputs=[signed, direction, reference, eps_a],
            output_shapes=[signed.shape], output_dtypes=[signed.dtype],
            grid=(int(signed.size), 1, 1), threadgroup=(256, 1, 1),
        )
        _record_pending("phase", "forward", out)
        return out

    @fn.vjp
    def _vjp(primals, cotangent, output):
        signed, direction, reference = primals
        cot = _raw_primal(cotangent)
        eps_a = mx.array([eps_value], dtype=mx.float32)
        (g_signed,) = _kernel("phase_vjp")(
            inputs=[signed, direction, reference, eps_a, cot],
            output_shapes=[signed.shape], output_dtypes=[signed.dtype],
            grid=(int(signed.size), 1, 1), threadgroup=(256, 1, 1),
        )
        # direction/reference are theta-independent provider maps. Preserve complete diagnostic
        # VJPs without doubling custom-kernel output traffic in every production training step.
        margin = mx.maximum(signed, 0.0)
        right = mx.pad(margin[:, :, 1:], [(0, 0), (0, 0), (0, 1)])
        down = mx.pad(margin[:, 1:, :], [(0, 0), (0, 1), (0, 0)])
        partner = mx.where(direction < 0.5, right, down)
        tie = margin / (margin + partner + eps_value)
        g_direction = mx.zeros_like(direction)
        g_reference = -2.0 * (tie - reference) * cot
        grads = (g_signed, g_direction, g_reference)
        _record_pending("phase", "backward", g_signed)
        return tuple(grads)

    return fn


def _make_temporal_metal():
    import mlx.core as mx

    @mx.custom_function
    def fn(g1, g0w, class_mask):
        shape = tuple(int(x) for x in g1.shape[:-1])
        (out,) = _kernel("temporal")(
            inputs=[g1, g0w, class_mask], output_shapes=[shape], output_dtypes=[g1.dtype],
            grid=(int(g1.size // 3), 1, 1), threadgroup=(256, 1, 1),
        )
        _record_pending("temporal", "forward", out)
        return out

    @fn.vjp
    def _vjp(primals, cotangent, output):
        g1, g0w, class_mask = primals
        cot = _raw_primal(cotangent)
        theta_grads = _kernel("temporal_vjp")(
            inputs=[g1, g0w, class_mask, cot],
            output_shapes=[g1.shape, g0w.shape],
            output_dtypes=[g1.dtype, g0w.dtype],
            grid=(int(g1.size // 3), 1, 1), threadgroup=(256, 1, 1),
        )
        # ``class_mask`` is a canonical constant/provider, never theta. Preserve a mathematically
        # complete custom-function contract for diagnostics with MLX's safe batch reduction; avoid
        # a risky atomic kernel solely to differentiate a non-trainable three-vector.
        g_class_mask = mx.sum(
            mx.square(g1 - g0w) * cot[..., None], axis=(0, 1, 2)
        )
        grads = (*theta_grads, g_class_mask)
        _record_pending("temporal", "backward", theta_grads)
        return grads

    return fn


def _chroma_metal_fn():
    global _CHROMA_FN
    if _CHROMA_FN is None:
        _CHROMA_FN = _make_chroma_metal()
    return _CHROMA_FN


def _phase_metal_fn(eps: float):
    key = float(eps)
    hit = _PHASE_FNS.get(key)
    if hit is None:
        hit = _make_phase_metal(key)
        _PHASE_FNS[key] = hit
    return hit


def _temporal_metal_fn():
    global _TEMPORAL_FN
    if _TEMPORAL_FN is None:
        _TEMPORAL_FN = _make_temporal_metal()
    return _TEMPORAL_FN


def chroma_squared_map(frame_rgb: Any, target_chroma: Any, *, use_metal: bool = True) -> Any:
    _require_rgb_batch("frame_rgb", frame_rgb)
    _require_rgb_batch("target_chroma", target_chroma)
    _require_equal_shape("chroma", frame_rgb, target_chroma)
    if use_metal and _metal_eligible(frame_rgb, target_chroma):
        _select_backend("chroma", "metal")
        # Mandatory fused forward and fused transpose/VJP. Receipt remains ``metal_planned``
        # until the faithful probe explicitly evaluates both legs.
        return _chroma_metal_fn()(frame_rgb, target_chroma)
    _select_backend("chroma", "mlx_reference")
    return chroma_squared_map_reference(frame_rgb, target_chroma)


def phase_squared_map(
    signed: Any, direction: Any, reference: Any, eps: float = 1e-6, *, use_metal: bool = True
) -> Any:
    _require_map_batch("signed", signed)
    _require_map_batch("direction", direction)
    _require_map_batch("reference", reference)
    _require_equal_shape("phase signed/direction", signed, direction)
    _require_equal_shape("phase signed/reference", signed, reference)
    if not math.isfinite(float(eps)) or not float(eps) > 0.0:
        raise ValueError(f"phase eps must be > 0, got {eps!r}")
    if use_metal and _metal_eligible(signed, direction, reference):
        _select_backend("phase", "metal")
        return _phase_metal_fn(eps)(signed, direction, reference)
    _select_backend("phase", "mlx_reference")
    return phase_squared_map_reference(signed, direction, reference, eps)


def temporal_squared_map(
    g1: Any, g0_warped: Any, class_mask: Any, *, use_metal: bool = True
) -> Any:
    _require_rgb_batch("g1", g1)
    _require_rgb_batch("g0_warped", g0_warped)
    _require_equal_shape("temporal", g1, g0_warped)
    if tuple(class_mask.shape) != (3,):
        raise ValueError(f"class_mask must have shape (3,), got {tuple(class_mask.shape)}")
    if use_metal and _metal_eligible(g1, g0_warped, class_mask):
        _select_backend("temporal", "metal")
        return _temporal_metal_fn()(g1, g0_warped, class_mask)
    _select_backend("temporal", "mlx_reference")
    return temporal_squared_map_reference(g1, g0_warped, class_mask)
