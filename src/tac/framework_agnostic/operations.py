# SPDX-License-Identifier: MIT
"""Canonical framework-agnostic operations dispatching per backend.

Per CLAUDE.md "MLX-FIRST NUMPY-PORTABLE INDIVIDUALLY-FRACTAL" 8th standing
directive + operator NON-NEGOTIABLE META directive 2026-05-28:
*"remmebr MLX first but agnostic portability via numpy and tinygrad like
primitives and helpers or decorators or whatever"*.

Sister of ``tac.substrates._shared.inflate_runtime`` (canonical inflate
helpers per Catalog #205) at the **training-time + bridge-contract surface**.
The inflate-time sister covers numpy-portable inflate primitives ≤200 LOC +
≤2 deps per HNeRV parity L4; THIS module covers the training-time +
quantization + brotli + bridge contract primitives.

Per CLAUDE.md "QAT pipeline" + "Quantizr archive contents" non-negotiables:
the canonical quantization helpers ``quantize_int8_per_channel`` +
``quantize_fp4_packed_nibbles`` produce byte-deterministic output across
backends so substrate trainers can fork the framework choice per Catalog
#205 sister discipline (MLX-LOCAL for $0 development; PyTorch CUDA for
contest-resolution promotion).

Per CLAUDE.md "Deterministic packet compiler" non-negotiable + Catalog #146:
the canonical brotli + inflate primitives produce byte-identical output
regardless of which backend trained the weights. The bridge contract MLX
state_dict → npz → ZIP-member → numpy inflate primitives is the structural
protection against framework-specific byte drift.

Per CLAUDE.md "Beauty, simplicity, and developer experience" + "tac stays
clean": each primitive is a thin dispatcher to the backend-specific
implementation; backend-specific implementations are deferred-imported per
Catalog #205 sister discipline so this module has zero hard framework
dependencies beyond numpy.

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" + Catalog #192/#317
non-negotiables: backend-specific tensors carry per-backend canonical
Provenance per Catalog #323 (MLX → non-promotable per Catalog #192; PyTorch
CUDA → contest-grade per Catalog #205 sister; numpy → diagnostic per
inflate-time contract).

Cross-references:
  * Catalog #205 — sister inflate-time helpers
  * Catalog #146 — contest-compliant inflate runtime contract
  * Catalog #323 — canonical Provenance umbrella
  * Catalog #287 — placeholder-rationale rejection sister discipline
  * tinygrad's ops.py (~600 LOC reference; this module is the lighter
    ~400 LOC initial contract scoped to the canonical primitives our
    substrates actually need)
"""
from __future__ import annotations

from typing import Any

from tac.framework_agnostic.backend import (
    Backend,
    BackendUnavailableError,
    select_backend,
)


# Canonical PyTorch FP4 codebook (unsigned E2M1) — sister of CLAUDE.md
# "Quantizr archive contents" verified empirical value
# ``[0,0.5,1,1.5,2,3,4,6]``. Used for QAT-compatible quantize_fp4 per
# CLAUDE.md "QAT pipeline" non-negotiable.
_FP4_CODEBOOK_UNSIGNED: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _resolve_backend(backend: Backend | None) -> Backend:
    """Resolve a possibly-None backend kwarg to a concrete Backend.

    None / AUTO defer to :func:`tac.framework_agnostic.backend.select_backend`
    via canonical priority.
    """
    if backend is None or backend is Backend.AUTO:
        return select_backend()
    return backend


# -----------------------------------------------------------------------------
# Canonical primitive: quantize_int8_per_channel
# -----------------------------------------------------------------------------


def quantize_int8_per_channel(
    tensor: Any,
    *,
    axis: int = 0,
    backend: Backend | None = None,
) -> tuple[Any, Any]:
    """Per-channel int8 symmetric quantization.

    Returns ``(int8_tensor, scale_tensor)`` such that
    ``dequantize_int8_per_channel(int8, scale) ≈ tensor`` within int8
    representability.

    Per CLAUDE.md "QAT pipeline" non-negotiable: per-channel scaling is the
    canonical PR101/PR102/PR103 medal-class quantization primitive. The
    scale tensor is float32 with shape ``(C,)`` where ``C`` = ``tensor.shape[axis]``.

    Per CLAUDE.md "Deterministic packet compiler" + Catalog #146: the
    quantization is byte-deterministic across backends (numpy reference
    implementation is the canonical oracle; backend-specific paths are
    structurally equivalent per the round-trip test in
    :mod:`tac.framework_agnostic.tests`).

    Args:
        tensor: FrameworkAgnosticTensor (torch.Tensor / mx.array / numpy.ndarray).
        axis: Dimension along which per-channel scales are computed.
            Default 0 (per-output-channel for conv2d weight shape (C_out, ...)).
        backend: Optional Backend override; defaults to canonical
            :func:`select_backend`.

    Returns:
        Tuple ``(int8_tensor, scale_tensor)``. Both tensors are on the
        same backend as the input.
    """
    b = _resolve_backend(backend)
    if b is Backend.NUMPY:
        return _quantize_int8_numpy(tensor, axis=axis)
    if b is Backend.PYTORCH:
        return _quantize_int8_pytorch(tensor, axis=axis)
    if b is Backend.MLX:
        return _quantize_int8_mlx(tensor, axis=axis)
    if b is Backend.TINYGRAD:
        return _quantize_int8_tinygrad(tensor, axis=axis)
    raise BackendUnavailableError(f"Unsupported Backend.{b.name} for quantize_int8_per_channel")


def dequantize_int8_per_channel(
    int8_tensor: Any,
    scale: Any,
    *,
    axis: int = 0,
    backend: Backend | None = None,
) -> Any:
    """Inverse of :func:`quantize_int8_per_channel`.

    Returns a float32 tensor whose per-channel scale is broadcast back per
    the canonical axis. Sister API for round-trip determinism check.
    """
    b = _resolve_backend(backend)
    if b is Backend.NUMPY:
        return _dequantize_int8_numpy(int8_tensor, scale, axis=axis)
    if b is Backend.PYTORCH:
        return _dequantize_int8_pytorch(int8_tensor, scale, axis=axis)
    if b is Backend.MLX:
        return _dequantize_int8_mlx(int8_tensor, scale, axis=axis)
    if b is Backend.TINYGRAD:
        return _dequantize_int8_tinygrad(int8_tensor, scale, axis=axis)
    raise BackendUnavailableError(f"Unsupported Backend.{b.name} for dequantize")


def _quantize_int8_numpy(tensor: Any, *, axis: int) -> tuple[Any, Any]:
    """numpy reference implementation — canonical oracle for byte determinism.

    Symmetric int8 (range -127..127 not -128..127 to preserve sign symmetry
    per PR101 medal-class convention).
    """
    import numpy as np  # noqa: PLC0415
    arr = np.asarray(tensor, dtype=np.float32)
    if axis < 0:
        axis = arr.ndim + axis
    # Per-channel abs-max reduction over all axes except `axis`.
    reduce_axes = tuple(i for i in range(arr.ndim) if i != axis)
    abs_max = np.max(np.abs(arr), axis=reduce_axes, keepdims=False)
    # Avoid div-by-zero on degenerate channels.
    scale = np.where(abs_max > 0, abs_max / 127.0, 1.0).astype(np.float32)
    # Broadcast scale back to tensor shape for division.
    broadcast_shape = [1] * arr.ndim
    broadcast_shape[axis] = arr.shape[axis]
    scale_broadcast = scale.reshape(broadcast_shape)
    int8 = np.round(arr / scale_broadcast).clip(-127, 127).astype(np.int8)
    return int8, scale


def _dequantize_int8_numpy(int8_tensor: Any, scale: Any, *, axis: int) -> Any:
    import numpy as np  # noqa: PLC0415
    int8 = np.asarray(int8_tensor, dtype=np.int8)
    sc = np.asarray(scale, dtype=np.float32)
    if axis < 0:
        axis = int8.ndim + axis
    broadcast_shape = [1] * int8.ndim
    broadcast_shape[axis] = int8.shape[axis]
    return int8.astype(np.float32) * sc.reshape(broadcast_shape)


def _quantize_int8_pytorch(tensor: Any, *, axis: int) -> tuple[Any, Any]:
    """PyTorch implementation; numpy-equivalent for byte determinism."""
    import torch  # noqa: PLC0415
    arr = torch.as_tensor(tensor, dtype=torch.float32)
    if axis < 0:
        axis = arr.dim() + axis
    reduce_axes = [i for i in range(arr.dim()) if i != axis]
    abs_max = arr.abs()
    for a in sorted(reduce_axes, reverse=True):
        abs_max = abs_max.amax(dim=a)
    scale = torch.where(abs_max > 0, abs_max / 127.0, torch.ones_like(abs_max))
    broadcast_shape = [1] * arr.dim()
    broadcast_shape[axis] = arr.shape[axis]
    scale_broadcast = scale.reshape(broadcast_shape)
    int8 = (arr / scale_broadcast).round().clamp(-127, 127).to(torch.int8)
    return int8, scale


def _dequantize_int8_pytorch(int8_tensor: Any, scale: Any, *, axis: int) -> Any:
    import torch  # noqa: PLC0415
    int8 = torch.as_tensor(int8_tensor, dtype=torch.int8)
    sc = torch.as_tensor(scale, dtype=torch.float32)
    if axis < 0:
        axis = int8.dim() + axis
    broadcast_shape = [1] * int8.dim()
    broadcast_shape[axis] = int8.shape[axis]
    return int8.to(torch.float32) * sc.reshape(broadcast_shape)


def _quantize_int8_mlx(tensor: Any, *, axis: int) -> tuple[Any, Any]:
    """MLX implementation per CLAUDE.md MLX-FIRST 8th standing directive."""
    import mlx.core as mx  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415  # canonical numpy oracle
    # MLX may not have the full reduce API; route through numpy oracle for
    # byte determinism and convert back. This is INTENTIONAL — the canonical
    # numpy oracle IS the byte-determinism guarantee per Catalog #146.
    arr_np = np.asarray(tensor, dtype=np.float32) if not isinstance(tensor, mx.array) else np.asarray(tensor)
    int8_np, scale_np = _quantize_int8_numpy(arr_np, axis=axis)
    return mx.array(int8_np), mx.array(scale_np)


def _dequantize_int8_mlx(int8_tensor: Any, scale: Any, *, axis: int) -> Any:
    import mlx.core as mx  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    int8_np = np.asarray(int8_tensor)
    sc_np = np.asarray(scale)
    result = _dequantize_int8_numpy(int8_np, sc_np, axis=axis)
    return mx.array(result)


def _quantize_int8_tinygrad(tensor: Any, *, axis: int) -> tuple[Any, Any]:
    """tinygrad implementation — deferred import; OPTIONAL backend."""
    try:
        from tinygrad import Tensor  # noqa: PLC0415
    except ImportError as exc:
        raise BackendUnavailableError(
            "Backend.TINYGRAD requested but tinygrad not installed; "
            "install via `uv pip install tinygrad`"
        ) from exc
    import numpy as np  # noqa: PLC0415
    arr_np = tensor.numpy() if isinstance(tensor, Tensor) else np.asarray(tensor, dtype=np.float32)
    int8_np, scale_np = _quantize_int8_numpy(arr_np, axis=axis)
    return Tensor(int8_np), Tensor(scale_np)


def _dequantize_int8_tinygrad(int8_tensor: Any, scale: Any, *, axis: int) -> Any:
    try:
        from tinygrad import Tensor  # noqa: PLC0415
    except ImportError as exc:
        raise BackendUnavailableError(
            "Backend.TINYGRAD requested but tinygrad not installed"
        ) from exc
    import numpy as np  # noqa: PLC0415
    int8_np = int8_tensor.numpy() if isinstance(int8_tensor, Tensor) else np.asarray(int8_tensor)
    sc_np = scale.numpy() if isinstance(scale, Tensor) else np.asarray(scale)
    result = _dequantize_int8_numpy(int8_np, sc_np, axis=axis)
    return Tensor(result)


# -----------------------------------------------------------------------------
# Canonical primitive: quantize_fp4_packed_nibbles
# -----------------------------------------------------------------------------


def quantize_fp4_packed_nibbles(
    tensor: Any,
    *,
    codebook: tuple[float, ...] | None = None,
    backend: Backend | None = None,
) -> tuple[Any, float]:
    """FP4 quantization to packed-nibble uint8 array per CLAUDE.md "QAT pipeline".

    Returns ``(packed_uint8, scale)`` where ``packed_uint8`` is a uint8 array
    with two FP4 nibbles per byte and ``scale`` is the canonical abs-max
    scale used to map the tensor into the unsigned-E2M1 codebook range.

    Per CLAUDE.md "Quantizr archive contents" verified empirical:
    ``codebook = [0, 0.5, 1, 1.5, 2, 3, 4, 6]`` (unsigned E2M1). Caller may
    pass a custom codebook (e.g., for signed-E2M1 variants) but the default
    is the canonical Quantizr / PR101 value.

    Args:
        tensor: FrameworkAgnosticTensor; flattened for nibble packing.
        codebook: Optional FP4 codebook (8 floats). Default: canonical
            Quantizr unsigned-E2M1.
        backend: Optional Backend override.

    Returns:
        Tuple ``(packed_uint8, scale)``. Length of packed_uint8 is
        ``(tensor.size + 1) // 2`` per nibble packing.
    """
    _ = _resolve_backend(backend)  # validate backend availability
    cb = codebook or _FP4_CODEBOOK_UNSIGNED
    if len(cb) != 8:
        raise ValueError(f"FP4 codebook must have 8 entries; got {len(cb)}")
    # Route through numpy oracle for byte determinism per Catalog #146.
    import numpy as np  # noqa: PLC0415
    arr = np.asarray(tensor, dtype=np.float32).flatten()
    abs_max = float(np.max(np.abs(arr))) if arr.size > 0 else 1.0
    if abs_max == 0:
        abs_max = 1.0
    scale = abs_max / max(cb)  # map abs_max to codebook upper bound
    scaled = np.abs(arr) / scale if scale > 0 else np.abs(arr)
    cb_arr = np.asarray(cb, dtype=np.float32)
    # Find nearest codebook entry per element (argmin over |scaled - cb|).
    diffs = np.abs(scaled[:, None] - cb_arr[None, :])
    indices = np.argmin(diffs, axis=1).astype(np.uint8)  # values 0..7
    # Pack pairs of 4-bit indices into uint8.
    if indices.size % 2 == 1:
        indices = np.concatenate([indices, np.zeros(1, dtype=np.uint8)])
    high = indices[0::2] << 4
    low = indices[1::2] & 0x0F
    packed = (high | low).astype(np.uint8)
    return packed, scale


# -----------------------------------------------------------------------------
# Canonical primitive: brotli_compress
# -----------------------------------------------------------------------------


def brotli_compress(byte_stream: bytes, *, quality: int = 11) -> bytes:
    """Canonical brotli compression per CLAUDE.md hard dependency.

    Per CLAUDE.md "Modal training image includes hard runtime deps" (Catalog
    #203/#224) + "Bit-level deconstruction and entropy discipline": brotli
    is the canonical entropy coder for PR101/PR102/PR103 medal-class
    archives. Quality 11 is the canonical default (slow but maximum
    compression; per Catalog #225 medal-class precedent).

    Note: this is framework-agnostic by construction (brotli is pure-C with
    Python bindings; no backend dispatch needed). Documented here for
    canonical API surface completeness.

    Args:
        byte_stream: Raw bytes to compress.
        quality: Brotli quality level (0..11). Default 11.

    Returns:
        Compressed bytes.
    """
    try:
        import brotli  # noqa: PLC0415
    except ImportError as exc:
        raise BackendUnavailableError(
            "brotli is a hard dependency per CLAUDE.md 'Modal training "
            "image includes hard runtime deps'; install via `uv pip install brotli`"
        ) from exc
    return brotli.compress(byte_stream, quality=quality)


__all__ = [
    "brotli_compress",
    "dequantize_int8_per_channel",
    "quantize_fp4_packed_nibbles",
    "quantize_int8_per_channel",
]
