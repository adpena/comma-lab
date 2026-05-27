# SPDX-License-Identifier: MIT
"""Canonical numpy-portable inflate BRIDGE — the decode-side foundation.

Per the 8th MLX-first + numpy-portable standing directive (2026-05-26):
*every substrate is MLX-first at TRAINING time AND numpy-portable at INFLATE
time (no torch / mlx / tensorflow / jax dependency at decode).* This module is
the single canonical surface every substrate ``inflate.py`` imports so the
shipped contest runtime tree carries ONLY numpy + brotli + PIL (within HNeRV
parity discipline L4: ``inflate.py`` ≤ 200 LOC, ≤ 2 external deps,
CUDA-or-CPU agnostic, reviewable in 30 seconds).

The bridge has four responsibilities (the canonical contract two sister
subagents build against):

1. **torch-AGNOSTIC state_dict serialization** — ``pack_state_dict_numpy`` /
   ``unpack_state_dict_numpy``. The root-cause finding of the
   ``class_shift_inflate_numpy_portability_audit`` was that every substrate
   archive stored decoder weights as ``brotli(pickle(torch_tensors))``. A
   torch-tensor pickle embeds ``torch._utils._rebuild_tensor_v2`` GLOBAL refs
   and CANNOT be unpickled without ``torch`` installed — so even a numpy-only
   inflate could not PARSE the archive. The numpy-native ``{key: ndarray}``
   blob (the ``coin_plus_plus`` CPP1-v2 pattern, parity 9e-6 vs torch) removes
   that structural blocker: it round-trips byte-stably with ZERO framework
   import.

2. **canonical decode primitives under a stable decode-side API** — the bridge
   wraps the torch/MLX-free primitives in
   ``tac.local_acceleration.pr95_hnerv_numpy_reference`` (``linear``,
   ``conv2d_nhwc``, ``bilinear_upsample_2x_nhwc``, ``sigmoid``, ``sin``,
   ``mean``) plus the NEW decode-only primitives needed by the class-shift
   decoders (``pixel_shuffle_2x_nhwc`` for boost_nerv/nirvana/atw, ``tanh`` +
   ``gru_cell_numpy`` for z5 autoregression, ``film_modulate_numpy`` for
   FiLM-conditioned coord-MLPs, ``relu``/``gelu`` activations, ``conv2d_numpy``
   + ``bilinear_resize_nhwc`` stable aliases). Substrate inflates import ONE
   place so the primitive set cannot drift across the 4 class-shift blockers +
   18 PACT-NeRV inflates.

3. **AST portability verifier** — ``assert_inflate_is_numpy_portable`` raises
   on any ``import torch`` / ``import mlx`` (and sister framework imports) in a
   shipped ``inflate.py``. This is the runtime sister of the STRICT preflight
   gate the harness stubs; consolidating it here means every substrate +
   emitter can call the same fail-closed check before claiming portability.

4. **canonical numpy-portable runtime emitter** —
   ``write_numpy_portable_contest_runtime`` is the torch-free sister of
   ``tac.substrates._shared.pact_nerv_full_main.write_contest_runtime``. It
   vendors ONLY numpy/PIL + the substrate's own ``archive.py`` + ``inflate.py``
   (NOT a torch ``architecture.py``) and self-verifies the emitted tree carries
   no framework import (Catalog #295 self-containment + the 8th directive).

Design principles (CLAUDE.md "Beauty, simplicity, and developer experience"):
fully typed, docstring'd, fail-closed on malformed blobs with clear errors,
deterministic + byte-stable, NO /tmp paths, NO scorer load, separation of
concerns (serialization vs primitives vs verification vs emission are distinct
public surfaces). Reusable + composable + OSS-grade.

Cross-references:
- ``src/tac/substrates/coin_plus_plus/{archive.py,inflate.py}`` (the
  proof-of-pattern; parity 9e-6) — these predate the bridge and should migrate
  to ``pack_state_dict_numpy`` / ``unpack_state_dict_numpy`` when next touched.
- ``.omx/research/class_shift_inflate_numpy_portability_audit_20260527T161212Z.md``
  (the per-decoder port spec).
- ``tac.local_acceleration.pr95_hnerv_numpy_reference`` (the torch/MLX-free
  primitive source the bridge re-exports).
- HNeRV parity discipline L4 (inflate ≤ 200 LOC / ≤ 2 deps) + Catalog
  #146 (contest runtime contract) + #205 (numpy is device-free; no MPS fork) +
  #295 (PYTHONPATH self-containment) + #367 (raw-byte fail-closed) + #369
  (consume real trained weights, not synthetic frame base).
"""

from __future__ import annotations

import ast
import io
import struct
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

# Re-export the canonical torch/MLX-free numpy reference primitives under the
# bridge's stable decode-side API. Substrate inflates import these from HERE so
# there is one canonical primitive surface (no per-substrate re-implementation
# drift). The underlying source is torch-free + mlx-free by construction.
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

# --------------------------------------------------------------------------- #
# Serialization grammar
# --------------------------------------------------------------------------- #

# The numpy-native state_dict blob format. Self-describing (key names + shapes
# + dtype) so the inflate-time reader needs NEITHER torch NOR ``np.load`` with
# ``allow_pickle`` (which is itself a deserialization hazard). Layout (before
# the optional brotli envelope)::
#
#     MAGIC(4)        b"NPSD"  numpy-portable state_dict
#     VERSION(1)      u8       schema version (currently 1)
#     DTYPE_CODE(1)   u8       0=float16, 1=float32, 2=int8, 3=uint8, 4=int16,
#                              5=int32, 6=float64, 7=bool
#     NUM_ENTRIES(4)  u32 LE
#     repeat NUM_ENTRIES:
#         KEY_LEN(2)  u16 LE
#         KEY         utf-8 bytes
#         NDIM(1)     u8
#         NDIM x u32 LE   shape dims
#         DATA        prod(shape) raw values (little-endian, C-order, DTYPE_CODE)
#
# A single shared DTYPE_CODE keeps the blob compact (the common case is a
# uniformly fp16 decoder state_dict); ``pack_state_dict_numpy`` casts every
# array to the requested dtype.
NPSD_MAGIC: bytes = b"NPSD"
NPSD_SCHEMA_VERSION: int = 1
_NPSD_HEADER_FMT: str = "<4sBBI"
_NPSD_HEADER_SIZE: int = struct.calcsize(_NPSD_HEADER_FMT)

# dtype <-> code mapping. Kept small + explicit (no eval of arbitrary dtype
# strings) so a malformed blob cannot smuggle an exotic dtype.
_DTYPE_BY_NAME: dict[str, int] = {
    "fp16": 0,
    "float16": 0,
    "fp32": 1,
    "float32": 1,
    "int8": 2,
    "uint8": 3,
    "int16": 4,
    "int32": 5,
    "fp64": 6,
    "float64": 6,
    "bool": 7,
}
_NP_DTYPE_BY_CODE: dict[int, np.dtype] = {
    0: np.dtype(np.float16),
    1: np.dtype(np.float32),
    2: np.dtype(np.int8),
    3: np.dtype(np.uint8),
    4: np.dtype(np.int16),
    5: np.dtype(np.int32),
    6: np.dtype(np.float64),
    7: np.dtype(np.bool_),
}
_ITEMSIZE_BY_CODE: dict[int, int] = {c: dt.itemsize for c, dt in _NP_DTYPE_BY_CODE.items()}

# Hard cap so a corrupt header (e.g. NUM_ENTRIES = 4 billion) fails closed fast
# rather than attempting a multi-GB allocation. Real decoder state_dicts have
# ~tens to low-hundreds of tensors.
_MAX_ENTRIES: int = 1 << 20
_MAX_NDIM: int = 8


class NumpyPortableStateDictError(ValueError):
    """Raised when a numpy-portable state_dict blob is malformed or unparseable.

    Distinct exception type so callers can ``except NumpyPortableStateDictError``
    to distinguish a corrupt-archive condition from an unrelated ``ValueError``
    in the decode forward pass (fail-closed per CLAUDE.md
    "Internal-consistency assertions").
    """


def _as_numpy(value: Any) -> np.ndarray:
    """Coerce a torch / MLX / numpy array to a numpy ndarray with NO framework import.

    Accepts anything exposing ``.detach().cpu().numpy()`` (torch),
    ``np.asarray(...)`` (MLX arrays are buffer-protocol / array-interface
    compatible), or a raw ndarray. The bridge never imports torch or mlx — it
    relies on duck-typing so a trainer that holds torch/MLX tensors can call
    ``pack_state_dict_numpy`` while the bridge module itself stays framework-free.
    """
    if isinstance(value, np.ndarray):
        return value
    # torch.Tensor path (duck-typed; no torch import in this module)
    detach = getattr(value, "detach", None)
    if callable(detach):
        try:
            t = detach()
            cpu = getattr(t, "cpu", None)
            if callable(cpu):
                t = cpu()
            np_fn = getattr(t, "numpy", None)
            if callable(np_fn):
                return np.asarray(np_fn())
        except Exception as exc:  # pragma: no cover - defensive
            raise NumpyPortableStateDictError(
                f"failed to convert tensor-like value to numpy: {exc}"
            ) from exc
    # MLX / array-interface path
    try:
        return np.asarray(value)
    except Exception as exc:
        raise NumpyPortableStateDictError(
            f"value is not numpy-convertible (type={type(value).__name__}): {exc}"
        ) from exc


def pack_state_dict_numpy(
    state_dict: Mapping[str, Any],
    *,
    dtype: str = "fp16",
) -> bytes:
    """Serialize a torch / MLX / numpy state_dict to a torch-AGNOSTIC numpy blob.

    The canonical bridge serializer: produces a self-describing ``{key: ndarray}``
    blob (key names + shapes + a single shared dtype) that round-trips
    byte-stably and is parseable with NO torch / mlx import. This is the
    structural fix for the ``brotli(pickle(torch_tensors))`` blocker — the
    output contains zero pickle, zero ``torch._utils._rebuild_tensor`` refs.

    Args:
        state_dict: mapping of parameter name -> tensor-like (torch.Tensor /
            mlx.array / np.ndarray). Order is preserved (Python dicts are
            insertion-ordered) so the blob is deterministic for a fixed input.
        dtype: target storage dtype name (default ``"fp16"`` — the canonical
            decoder weight storage that halves rate vs fp32 with negligible
            decode-parity loss). One of the keys of ``_DTYPE_BY_NAME``.

    Returns:
        The numpy-native state_dict blob (NOT brotli-compressed; the caller's
        archive grammar owns compression so the bridge stays composable — an
        archive that already brotli-wraps its sections does not double-compress).

    Raises:
        NumpyPortableStateDictError: on unknown dtype, non-string key, oversized
            key, or non-numpy-convertible value.
    """
    if dtype not in _DTYPE_BY_NAME:
        raise NumpyPortableStateDictError(
            f"unknown dtype {dtype!r}; expected one of {sorted(_DTYPE_BY_NAME)}"
        )
    dtype_code = _DTYPE_BY_NAME[dtype]
    np_dtype = _NP_DTYPE_BY_CODE[dtype_code]

    entries: list[tuple[bytes, np.ndarray]] = []
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise NumpyPortableStateDictError(
                f"state_dict keys must be str; got {type(key).__name__}"
            )
        key_bytes = key.encode("utf-8")
        if len(key_bytes) > 0xFFFF:
            raise NumpyPortableStateDictError(
                f"state_dict key too long ({len(key_bytes)} bytes > 65535): {key!r}"
            )
        # NB: np.ascontiguousarray PROMOTES a 0-dim scalar to shape (1,); use
        # np.asarray (preserves ndim==0) then enforce C-contiguity without
        # rank change so a scalar weight round-trips to shape ().
        arr = np.asarray(_as_numpy(value), dtype=np_dtype)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
            if arr.ndim == 1 and _as_numpy(value).ndim == 0:  # pragma: no cover
                arr = arr.reshape(())
        if arr.ndim > _MAX_NDIM:
            raise NumpyPortableStateDictError(
                f"array for key {key!r} has ndim {arr.ndim} > {_MAX_NDIM}"
            )
        entries.append((key_bytes, arr))

    if len(entries) > _MAX_ENTRIES:
        raise NumpyPortableStateDictError(
            f"too many entries ({len(entries)} > {_MAX_ENTRIES})"
        )

    buf = io.BytesIO()
    buf.write(struct.pack(_NPSD_HEADER_FMT, NPSD_MAGIC, NPSD_SCHEMA_VERSION, dtype_code, len(entries)))
    for key_bytes, arr in entries:
        buf.write(struct.pack("<H", len(key_bytes)))
        buf.write(key_bytes)
        buf.write(struct.pack("<B", arr.ndim))
        for dim in arr.shape:
            buf.write(struct.pack("<I", int(dim)))
        buf.write(arr.tobytes(order="C"))
    return buf.getvalue()


def unpack_state_dict_numpy(blob: bytes) -> dict[str, np.ndarray]:
    """Pure-numpy inverse of ``pack_state_dict_numpy`` — ZERO framework import.

    Fail-closed on every malformed condition (bad magic, unknown version,
    truncated payload, oversized counts) so a corrupt archive cannot silently
    decode to garbage. The returned arrays are the EXACT storage dtype the blob
    was packed with (e.g. ``np.float16``); decode forward passes that need fp32
    accumulation should cast via ``to_float32`` (the bridge primitive).

    Args:
        blob: the bytes produced by ``pack_state_dict_numpy``.

    Returns:
        ``{key: np.ndarray}`` in the blob's insertion order.

    Raises:
        NumpyPortableStateDictError: on any structural malformation.
    """
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise NumpyPortableStateDictError(
            f"blob must be bytes-like; got {type(blob).__name__}"
        )
    mv = memoryview(bytes(blob))
    if len(mv) < _NPSD_HEADER_SIZE:
        raise NumpyPortableStateDictError(
            f"blob too short ({len(mv)} bytes; need >= {_NPSD_HEADER_SIZE})"
        )
    magic, version, dtype_code, num_entries = struct.unpack_from(_NPSD_HEADER_FMT, mv, 0)
    if magic != NPSD_MAGIC:
        raise NumpyPortableStateDictError(
            f"bad magic {magic!r} (expected {NPSD_MAGIC!r})"
        )
    if version != NPSD_SCHEMA_VERSION:
        raise NumpyPortableStateDictError(
            f"unsupported schema version {version} (expected {NPSD_SCHEMA_VERSION})"
        )
    if dtype_code not in _NP_DTYPE_BY_CODE:
        raise NumpyPortableStateDictError(f"unknown dtype code {dtype_code}")
    if num_entries > _MAX_ENTRIES:
        raise NumpyPortableStateDictError(
            f"num_entries {num_entries} exceeds cap {_MAX_ENTRIES} (likely corrupt)"
        )
    np_dtype = _NP_DTYPE_BY_CODE[dtype_code]
    itemsize = _ITEMSIZE_BY_CODE[dtype_code]

    off = _NPSD_HEADER_SIZE
    out: dict[str, np.ndarray] = {}
    total = len(mv)
    for _ in range(num_entries):
        if off + 2 > total:
            raise NumpyPortableStateDictError("truncated blob reading key_len")
        (key_len,) = struct.unpack_from("<H", mv, off)
        off += 2
        if off + key_len > total:
            raise NumpyPortableStateDictError("truncated blob reading key")
        key = bytes(mv[off : off + key_len]).decode("utf-8")
        off += key_len
        if off + 1 > total:
            raise NumpyPortableStateDictError("truncated blob reading ndim")
        (ndim,) = struct.unpack_from("<B", mv, off)
        off += 1
        if ndim > _MAX_NDIM:
            raise NumpyPortableStateDictError(f"ndim {ndim} > {_MAX_NDIM} for key {key!r}")
        shape: list[int] = []
        count = 1
        for _d in range(ndim):
            if off + 4 > total:
                raise NumpyPortableStateDictError("truncated blob reading shape dim")
            (dim,) = struct.unpack_from("<I", mv, off)
            off += 4
            shape.append(int(dim))
            count *= int(dim)
        nbytes = count * itemsize
        if off + nbytes > total:
            raise NumpyPortableStateDictError(
                f"truncated blob reading data for key {key!r} "
                f"(need {nbytes} bytes, have {total - off})"
            )
        arr = np.frombuffer(mv[off : off + nbytes], dtype=np_dtype).copy()
        off += nbytes
        out[key] = arr.reshape(shape) if shape else arr.reshape(())
    if off != total:
        raise NumpyPortableStateDictError(
            f"trailing bytes after {num_entries} entries (offset {off} != length {total})"
        )
    return out


# --------------------------------------------------------------------------- #
# Decode primitives — stable decode-side API
# --------------------------------------------------------------------------- #
#
# The canonical reference primitives (linear / conv2d_nhwc /
# bilinear_upsample_2x_nhwc / sigmoid / sin / mean / to_float32 / kahan_mean)
# are re-exported above. Below are stable aliases under the names the prompt
# contract uses + the NEW decode-only primitives needed by the class-shift
# decoders (pixel_shuffle, FiLM, GRU, tanh, activations). All numpy-native; no
# torch / mlx import.


def conv2d_numpy(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    padding: int = 0,
) -> np.ndarray:
    """Stable decode-side alias for :func:`conv2d_nhwc` (NHWC 2D conv, fp32 accum)."""
    return conv2d_nhwc(x, weight, bias, padding=padding)


def bilinear_resize_nhwc(
    x: np.ndarray,
    *,
    target_h: int,
    target_w: int,
    align_corners: bool = False,
) -> np.ndarray:
    """Generalized bilinear resize for NHWC tensors to arbitrary ``(target_h, target_w)``.

    Torch/MLX-free decode-side sister of
    ``tac.local_acceleration.pr95_hnerv_mlx.bilinear_resize_nhwc``. CANONICAL
    default ``align_corners=False`` (PyTorch ``F.interpolate`` default; the
    DreamerV3 align_corners=True drift anchor documents a 24.34 max-abs gap when
    this is wrong). Each output pixel maps to input via::

        align_corners=False: src = (dst + 0.5) * (in / out) - 0.5
        align_corners=True:  src = dst * (in - 1) / (out - 1)

    then clamped to ``[0, in-1]`` and bilinearly interpolated. fp32 accumulation;
    ≤ 1e-5 parity vs PyTorch.

    Args:
        x: input shape ``(N, H, W, C)``.
        target_h, target_w: output spatial extent.
        align_corners: PyTorch ``F.interpolate`` semantics (default False).

    Returns:
        Output shape ``(N, target_h, target_w, C)``.
    """
    x32 = to_float32(x)
    if x32.ndim != 4:
        raise ValueError(f"bilinear_resize_nhwc expects NHWC 4-D; got shape {x32.shape}")
    n, h, w, c = x32.shape
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"target dims must be positive; got ({target_h}, {target_w})")

    dst_y = np.arange(target_h, dtype=np.float32)
    dst_x = np.arange(target_w, dtype=np.float32)
    if align_corners:
        src_y = dst_y * ((h - 1) / (target_h - 1)) if target_h > 1 else np.zeros_like(dst_y)
        src_x = dst_x * ((w - 1) / (target_w - 1)) if target_w > 1 else np.zeros_like(dst_x)
    else:
        src_y = (dst_y + 0.5) * (h / target_h) - 0.5
        src_x = (dst_x + 0.5) * (w / target_w) - 0.5
    src_y = np.clip(src_y, 0.0, h - 1.0)
    src_x = np.clip(src_x, 0.0, w - 1.0)

    y0 = np.floor(src_y).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    x0 = np.floor(src_x).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    fy = src_y - y0.astype(np.float32)
    fx = src_x - x0.astype(np.float32)

    y0g, x0g = np.meshgrid(y0, x0, indexing="ij")
    y1g, x1g = np.meshgrid(y1, x1, indexing="ij")
    fyg, fxg = np.meshgrid(fy, fx, indexing="ij")

    w00 = ((1.0 - fxg) * (1.0 - fyg))[None, :, :, None]
    w01 = (fxg * (1.0 - fyg))[None, :, :, None]
    w10 = ((1.0 - fxg) * fyg)[None, :, :, None]
    w11 = (fxg * fyg)[None, :, :, None]

    p00 = x32[:, y0g, x0g, :]
    p01 = x32[:, y0g, x1g, :]
    p10 = x32[:, y1g, x0g, :]
    p11 = x32[:, y1g, x1g, :]
    return w00 * p00 + w01 * p01 + w10 * p10 + w11 * p11


def pixel_shuffle_2x_nhwc(x: np.ndarray) -> np.ndarray:
    """PyTorch ``nn.PixelShuffle(2)`` in NHWC layout (4× channel -> 2× spatial).

    Used by the boost_nerv / nirvana / atw_codec_v1 convolutional decoders.
    PyTorch ``PixelShuffle`` is defined on NCHW: ``(N, C*r^2, H, W) ->
    (N, C, H*r, W*r)`` where the ``r^2`` channel block is unfolded as
    ``(r, r)`` spatial sub-pixels in **row-major (rh, rw)** order. This numpy
    NHWC port reproduces that EXACT element ordering so it is byte-stable vs the
    torch reference. ``r = 2`` (the canonical NeRV-family upsample factor).

    Args:
        x: input shape ``(N, H, W, 4*C)`` (NHWC; channels-last).

    Returns:
        Output shape ``(N, 2*H, 2*W, C)``.

    Raises:
        ValueError: if the channel count is not divisible by 4.
    """
    x32 = to_float32(x)
    if x32.ndim != 4:
        raise ValueError(f"pixel_shuffle_2x_nhwc expects NHWC 4-D; got shape {x32.shape}")
    r = 2
    n, h, w, c_in = x32.shape
    if c_in % (r * r) != 0:
        raise ValueError(
            f"pixel_shuffle_2x_nhwc: channel count {c_in} not divisible by {r * r}"
        )
    c_out = c_in // (r * r)
    # PyTorch NCHW PixelShuffle reshapes channel as (C_out, r, r). In NHWC the
    # channel axis carries the same (C_out, rh, rw) interleave, so:
    #   (N, H, W, C_out, r, r) -> transpose to (N, H, r_h, W, r_w, C_out)
    #   -> reshape (N, H*r, W*r, C_out)
    x6 = x32.reshape(n, h, w, c_out, r, r)
    x6 = np.transpose(x6, (0, 1, 4, 2, 5, 3))  # (N, H, rh, W, rw, C_out)
    return np.ascontiguousarray(x6).reshape(n, h * r, w * r, c_out)


def relu(x: np.ndarray) -> np.ndarray:
    """Element-wise ReLU; fp32."""
    return np.maximum(to_float32(x), 0.0)


def tanh(x: np.ndarray) -> np.ndarray:
    """Element-wise tanh; fp32 (GRU/LSTM gate activation for z5 autoregression)."""
    return np.tanh(to_float32(x))


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit (tanh approximation; PyTorch ``approximate='tanh'``)."""
    x32 = to_float32(x)
    c = np.float32(0.7978845608028654)  # sqrt(2/pi)
    return 0.5 * x32 * (1.0 + np.tanh(c * (x32 + 0.044715 * x32 * x32 * x32)))


def film_modulate_numpy(
    h: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """FiLM modulation: ``gamma * h + beta`` with broadcasting.

    The canonical feature-wise linear modulation (Perez et al. 2018) used by
    FiLM-conditioned coord-MLP decoders (coin_plus_plus pattern). ``gamma`` /
    ``beta`` are per-feature vectors; ``h`` is ``(..., features)``. Broadcasts
    the modulation across the leading (spatial / batch) axes. fp32.

    Args:
        h: pre-modulation features, shape ``(..., F)``.
        gamma: per-feature scale, shape ``(F,)`` or broadcastable.
        beta: per-feature shift, shape ``(F,)`` or broadcastable.

    Returns:
        ``gamma * h + beta`` in fp32, shape of ``h``.
    """
    h32 = to_float32(h)
    g32 = to_float32(gamma)
    b32 = to_float32(beta)
    return g32 * h32 + b32


def gru_cell_numpy(
    x: np.ndarray,
    h_prev: np.ndarray,
    *,
    weight_ih: np.ndarray,
    weight_hh: np.ndarray,
    bias_ih: np.ndarray | None = None,
    bias_hh: np.ndarray | None = None,
) -> np.ndarray:
    """One step of a PyTorch ``nn.GRUCell`` in pure numpy (z5 autoregression).

    Reproduces ``torch.nn.GRUCell`` EXACTLY, including the canonical gate split
    and the ``r * (W_hn @ h + b_hn)`` reset-applied-inside-candidate form::

        gates_x = x @ W_ih.T + b_ih     # (..., 3*hidden): [r | z | n] blocks
        gates_h = h @ W_hh.T + b_hh     # (..., 3*hidden): [r | z | n] blocks
        r = sigmoid(gates_x[r] + gates_h[r])
        z = sigmoid(gates_x[z] + gates_h[z])
        n = tanh(gates_x[n] + r * gates_h[n])     # reset INSIDE candidate
        h_new = (1 - z) * n + z * h_prev

    PyTorch concatenates the gate weights in **(reset, update, new)** order;
    this port slices the ``3*hidden`` axis in that order so the weights map
    byte-stably.

    Args:
        x: input, shape ``(..., input_size)``.
        h_prev: previous hidden state, shape ``(..., hidden_size)``.
        weight_ih: shape ``(3*hidden_size, input_size)`` (PyTorch canonical).
        weight_hh: shape ``(3*hidden_size, hidden_size)``.
        bias_ih, bias_hh: optional, shape ``(3*hidden_size,)``.

    Returns:
        New hidden state, shape ``(..., hidden_size)``.
    """
    x32 = to_float32(x)
    h32 = to_float32(h_prev)
    gi = linear(x32, weight_ih, bias_ih)  # (..., 3H)
    gh = linear(h32, weight_hh, bias_hh)  # (..., 3H)
    hidden = h32.shape[-1]
    if gi.shape[-1] != 3 * hidden or gh.shape[-1] != 3 * hidden:
        raise ValueError(
            f"gru_cell_numpy gate width mismatch: gi={gi.shape[-1]}, gh={gh.shape[-1]}, "
            f"expected 3*hidden={3 * hidden}"
        )
    i_r, i_z, i_n = gi[..., 0:hidden], gi[..., hidden : 2 * hidden], gi[..., 2 * hidden : 3 * hidden]
    h_r, h_z, h_n = gh[..., 0:hidden], gh[..., hidden : 2 * hidden], gh[..., 2 * hidden : 3 * hidden]
    r = sigmoid(i_r + h_r)
    z = sigmoid(i_z + h_z)
    n = np.tanh(i_n + r * h_n)
    return (1.0 - z) * n + z * h32


# --------------------------------------------------------------------------- #
# Raw-output writer (contest .raw lowering, numpy-native)
# --------------------------------------------------------------------------- #

CAMERA_HW: tuple[int, int] = (874, 1164)
"""Contest output resolution (H, W). One raw uint8 RGB frame = H*W*3 bytes."""

CONTEST_NUM_FRAMES: int = 1200
CONTEST_RAW_BYTES_PER_VIDEO: int = CAMERA_HW[0] * CAMERA_HW[1] * CONTEST_NUM_FRAMES * 3
"""Per-video raw byte count: 874 * 1164 * 1200 * 3 = 3,662,409,600 (Catalog #367)."""


def write_rgb_pair_to_raw_numpy(
    fh: Any,
    rgb_0: np.ndarray,
    rgb_1: np.ndarray,
    *,
    input_range: str = "unit",
) -> int:
    """Append one rendered frame-pair to an open contest ``.raw`` file (numpy-native).

    Torch-free sister of
    ``tac.substrates._shared.inflate_runtime.write_rgb_pair_to_raw``. Accepts
    NHWC ``(1, H, W, 3)`` numpy frames (the numpy decode convention; the torch
    helper used NCHW). Resizes to contest ``CAMERA_HW`` via the bridge bilinear
    resize when needed, clamps + rounds to uint8, and writes row-major C-order
    bytes. Catalog #367: the per-video byte total must equal
    ``CONTEST_RAW_BYTES_PER_VIDEO``; the caller (inflate) asserts that after all
    pairs are written (this helper writes exactly 2 frames).

    Args:
        fh: binary file handle opened for write/append.
        rgb_0, rgb_1: NHWC tensors shaped ``(1, H, W, 3)``.
        input_range: ``"unit"`` for ``[0, 1]`` inputs, ``"byte"`` for ``[0, 255]``.

    Returns:
        Number of frames written (always 2 for valid inputs).
    """
    a0 = to_float32(rgb_0)
    a1 = to_float32(rgb_1)
    if a0.shape != a1.shape or a0.ndim != 4 or a0.shape[0] != 1 or a0.shape[3] != 3:
        raise ValueError(
            "write_rgb_pair_to_raw_numpy expects two NHWC tensors shaped (1, H, W, 3); "
            f"got {a0.shape} and {a1.shape}"
        )
    frames = np.concatenate([a0, a1], axis=0)  # (2, H, W, 3)
    if input_range == "unit":
        frames = frames * 255.0
    elif input_range != "byte":
        raise ValueError(f"input_range must be 'unit' or 'byte'; got {input_range!r}")
    if (frames.shape[1], frames.shape[2]) != CAMERA_HW:
        frames = bilinear_resize_nhwc(
            frames, target_h=CAMERA_HW[0], target_w=CAMERA_HW[1], align_corners=False
        )
    frames_u8 = np.clip(np.round(frames), 0.0, 255.0).astype(np.uint8)
    fh.write(np.ascontiguousarray(frames_u8).tobytes(order="C"))
    return int(frames_u8.shape[0])


# --------------------------------------------------------------------------- #
# AST portability verifier
# --------------------------------------------------------------------------- #

# Frameworks FORBIDDEN at inflate time per the 8th MLX-first directive. Decode
# must be numpy-portable; importing any of these makes the runtime tree
# non-portable.
FORBIDDEN_INFLATE_FRAMEWORKS: tuple[str, ...] = (
    "torch",
    "mlx",
    "tensorflow",
    "jax",
    "jaxlib",
)


class InflateNotNumpyPortableError(ValueError):
    """Raised when an inflate.py imports a forbidden framework (torch/mlx/...)."""


def _forbidden_root(module_name: str) -> str | None:
    """Return the forbidden top-level package if ``module_name`` imports one."""
    root = module_name.split(".", 1)[0]
    return root if root in FORBIDDEN_INFLATE_FRAMEWORKS else None


def find_forbidden_framework_imports(source: str) -> list[tuple[int, str]]:
    """Return ``(lineno, framework_root)`` for every forbidden framework import.

    Pure AST walk (no import / exec of the target). Detects both
    ``import torch`` / ``import torch.nn`` (Import nodes) and
    ``from torch import ...`` (ImportFrom nodes). Relative imports
    (``from . import x``) are ignored. A syntactically invalid source raises
    ``SyntaxError`` (the caller decides whether that is fatal).
    """
    tree = ast.parse(source)
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _forbidden_root(alias.name)
                if root is not None:
                    hits.append((node.lineno, root))
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative import; not a framework package
            if node.module is None:
                continue
            root = _forbidden_root(node.module)
            if root is not None:
                hits.append((node.lineno, root))
    return hits


def assert_inflate_is_numpy_portable(inflate_path: Any) -> None:
    """Raise ``InflateNotNumpyPortableError`` if ``inflate_path`` imports a forbidden framework.

    The canonical fail-closed verifier the harness + emitter + substrate authors
    call to PROVE a shipped ``inflate.py`` is numpy-portable. Consolidates the
    forbidden-framework AST scan into one place so every surface checks the same
    contract.

    Args:
        inflate_path: pathlib.Path or str to the inflate.py to verify.

    Raises:
        InflateNotNumpyPortableError: if any forbidden framework import is found,
            with the line numbers + framework names in the message.
        FileNotFoundError: if the file does not exist.
    """
    path = Path(inflate_path)
    source = path.read_text(encoding="utf-8")
    hits = find_forbidden_framework_imports(source)
    if hits:
        detail = ", ".join(f"line {ln}: import {fw}" for ln, fw in hits)
        raise InflateNotNumpyPortableError(
            f"{path}: inflate.py is NOT numpy-portable — forbidden framework "
            f"import(s) found ({detail}). Per the 8th MLX-first standing directive, "
            f"inflate must be numpy/PIL-portable (no {'/'.join(FORBIDDEN_INFLATE_FRAMEWORKS)}). "
            f"Store weights via pack_state_dict_numpy and decode via the bridge primitives."
        )


# --------------------------------------------------------------------------- #
# Canonical numpy-portable runtime emitter
# --------------------------------------------------------------------------- #


def _build_numpy_portable_inflate_sh(substrate_pkg_name: str) -> str:
    """Return the contest-compliant ``inflate.sh`` (Catalog #146 3-arg contract)."""
    return (
        "#!/usr/bin/env bash\n"
        f"# {substrate_pkg_name} numpy-portable contest inflate "
        "(canonical numpy-portable inflate bridge)\n"
        "# Contract: $1=archive_dir $2=output_dir $3=file_list\n"
        "set -euo pipefail\n"
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'DATA_DIR="$1"\n'
        'OUTPUT_DIR="$2"\n'
        'FILE_LIST="$3"\n'
        'mkdir -p "$OUTPUT_DIR"\n'
        'exec "${PYTHON:-python3}" "$HERE/inflate.py" '
        '"$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"\n'
    )


def write_numpy_portable_contest_runtime(
    submission_dir: Any,
    *,
    substrate_pkg_name: str,
    repo_root: Any,
    runtime_module_files: tuple[str, ...] = ("archive.py", "inflate.py"),
    extra_vendor_files: tuple[str, ...] = (),
    verify_portable: bool = True,
) -> None:
    """Emit a torch-FREE contest runtime tree (numpy/PIL-portable).

    The canonical sister of
    ``tac.substrates._shared.pact_nerv_full_main.write_contest_runtime``, but it
    vendors ONLY the substrate's ``archive.py`` + ``inflate.py`` (NOT a torch
    ``architecture.py``) plus this bridge module, and self-verifies the emitted
    ``inflate.py`` carries no forbidden framework import. The runtime-bundler all
    numpy-portable substrates call.

    Emitted tree (Catalog #295 self-containment)::

        submission_dir/
          inflate.sh                      # 3-arg contract, set -euo pipefail
          inflate.py                      # numpy-only thin CLI -> inflate_one_video
          src/tac/__init__.py             # ""
          src/tac/substrates/__init__.py  # ""
          src/tac/substrates/<pkg>/__init__.py
          src/tac/substrates/<pkg>/archive.py   # numpy-native parser (no torch)
          src/tac/substrates/<pkg>/inflate.py   # numpy decode (no torch)
          src/tac/substrates/_shared/__init__.py
          src/tac/substrates/_shared/numpy_portable_inflate.py  # THIS bridge
          src/tac/local_acceleration/__init__.py
          src/tac/local_acceleration/pr95_hnerv_numpy_reference.py  # primitives

    The vendored ``inflate.py`` puts ``HERE/src`` on ``sys.path`` so the import
    resolves with an EMPTY PYTHONPATH (Catalog #295) and carries only numpy +
    brotli + PIL deps (HNeRV parity L4 ≤ 2 external deps; numpy/brotli are the
    two, PIL only for PNG-mode substrates).

    Args:
        submission_dir: pathlib.Path; the contest submission directory.
        substrate_pkg_name: e.g. ``"coin_plus_plus"`` (the substrates/ subdir).
        repo_root: pathlib.Path of the repo root (to locate source modules).
        runtime_module_files: substrate package modules to vendor (default
            ``archive.py`` + ``inflate.py``; NO ``architecture.py`` because the
            numpy decode lives in inflate.py + the bridge primitives).
        extra_vendor_files: additional per-substrate package files to vendor
            (e.g. a substrate-specific ``numpy_reference.py``).
        verify_portable: if True (default), run
            ``assert_inflate_is_numpy_portable`` on the emitted + vendored
            inflate.py and raise if any forbidden import slipped in (fail-closed
            self-protection).

    Raises:
        InflateNotNumpyPortableError: if ``verify_portable`` and the vendored
            inflate.py imports a forbidden framework.
        FileNotFoundError: if a declared source module is missing.
    """
    import shutil

    submission_dir = Path(submission_dir)
    repo_root = Path(repo_root)
    submission_dir.mkdir(parents=True, exist_ok=True)

    src_root = submission_dir / "src"
    runtime_pkg = src_root / "tac" / "substrates" / substrate_pkg_name
    shared_pkg = src_root / "tac" / "substrates" / "_shared"
    accel_pkg = src_root / "tac" / "local_acceleration"
    runtime_pkg.mkdir(parents=True, exist_ok=True)
    shared_pkg.mkdir(parents=True, exist_ok=True)
    accel_pkg.mkdir(parents=True, exist_ok=True)

    for pkg_init in (
        src_root / "tac" / "__init__.py",
        src_root / "tac" / "substrates" / "__init__.py",
        runtime_pkg / "__init__.py",
        shared_pkg / "__init__.py",
        accel_pkg / "__init__.py",
    ):
        pkg_init.write_text("", encoding="utf-8")

    # Vendor the substrate's numpy-native modules.
    substrate_src = repo_root / "src" / "tac" / "substrates" / substrate_pkg_name
    for name in tuple(runtime_module_files) + tuple(extra_vendor_files):
        src_file = substrate_src / name
        if not src_file.is_file():
            raise FileNotFoundError(
                f"declared runtime module not found: {src_file} "
                f"(substrate_pkg_name={substrate_pkg_name!r})"
            )
        shutil.copy2(src_file, runtime_pkg / name)

    # Vendor THIS bridge + the canonical numpy primitive reference (the only two
    # _shared / local_acceleration files the numpy decode path needs).
    bridge_src = repo_root / "src" / "tac" / "substrates" / "_shared" / "numpy_portable_inflate.py"
    primitives_src = repo_root / "src" / "tac" / "local_acceleration" / "pr95_hnerv_numpy_reference.py"
    for src_file, dst in (
        (bridge_src, shared_pkg / "numpy_portable_inflate.py"),
        (primitives_src, accel_pkg / "pr95_hnerv_numpy_reference.py"),
    ):
        if not src_file.is_file():
            raise FileNotFoundError(f"canonical bridge dependency not found: {src_file}")
        shutil.copy2(src_file, dst)

    # Emit inflate.sh (3-arg contract).
    (submission_dir / "inflate.sh").write_text(
        _build_numpy_portable_inflate_sh(substrate_pkg_name), encoding="utf-8"
    )
    (submission_dir / "inflate.sh").chmod(0o755)

    # Emit the thin numpy-only inflate.py CLI delegating to the vendored
    # substrate inflate_one_video. Note: the substrate's OWN vendored inflate.py
    # may already carry a main_cli; this top-level shim is the canonical Catalog
    # #146 entry point and stays framework-free.
    inflate_import_line = (
        f"from tac.substrates.{substrate_pkg_name}.inflate import inflate_one_video"
    )
    inflate_py = (
        "#!/usr/bin/env python\n"
        f'"""{substrate_pkg_name} numpy-portable contest inflate runtime.\n'
        "\n"
        "Reads archive_dir/0.bin via the vendored numpy-native parser, then for\n"
        "each base in file_list decodes frames under output_dir/<base>/.\n"
        "numpy/PIL-portable: no torch / mlx import (8th MLX-first directive).\n"
        '"""\n'
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "HERE = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(HERE / 'src'))\n"
        f"{inflate_import_line}\n"
        "\n"
        "\n"
        "def main() -> int:\n"
        "    if len(sys.argv) != 4:\n"
        "        print('usage: inflate.py <archive_dir> <output_dir> <file_list>',\n"
        "              file=sys.stderr)\n"
        "        return 2\n"
        "    archive_dir = Path(sys.argv[1])\n"
        "    output_dir = Path(sys.argv[2])\n"
        "    file_list_path = Path(sys.argv[3])\n"
        "    archive_bytes = (archive_dir / '0.bin').read_bytes()\n"
        "    for line in file_list_path.read_text(encoding='utf-8').splitlines():\n"
        "        line = line.strip()\n"
        "        if not line:\n"
        "            continue\n"
        "        base = line.rsplit('.', 1)[0]\n"
        "        inflate_one_video(archive_bytes, output_dir / base)\n"
        "    return 0\n"
        "\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(main())\n"
    )
    (submission_dir / "inflate.py").write_text(inflate_py, encoding="utf-8")

    if verify_portable:
        # Self-protection (fail-closed): EVERY vendored .py in the emitted tree
        # must be numpy-portable. A module's top-level ``import torch`` crashes a
        # torch-free worker at import time regardless of whether its functions
        # are called, so the whole tree — not just the entry inflate.py — must be
        # framework-free. This is the structural enforcement of the 8th MLX-first
        # directive at emission time.
        for py in submission_dir.rglob("*.py"):
            assert_inflate_is_numpy_portable(py)


# A registry of the decode primitives the bridge exposes (for introspection /
# documentation / the AST verifier's allowlist). Maps stable decode-side name
# to the callable.
DECODE_PRIMITIVES: dict[str, Callable[..., Any]] = {
    "to_float32": to_float32,
    "linear": linear,
    "conv2d_nhwc": conv2d_nhwc,
    "conv2d_numpy": conv2d_numpy,
    "bilinear_upsample_2x_nhwc": bilinear_upsample_2x_nhwc,
    "bilinear_resize_nhwc": bilinear_resize_nhwc,
    "pixel_shuffle_2x_nhwc": pixel_shuffle_2x_nhwc,
    "film_modulate_numpy": film_modulate_numpy,
    "gru_cell_numpy": gru_cell_numpy,
    "sigmoid": sigmoid,
    "sin": sin,
    "tanh": tanh,
    "relu": relu,
    "gelu": gelu,
    "mean": mean,
    "kahan_mean": kahan_mean,
}


__all__ = [
    # raw-output writer
    "CAMERA_HW",
    "CONTEST_NUM_FRAMES",
    "CONTEST_RAW_BYTES_PER_VIDEO",
    "DECODE_PRIMITIVES",
    # AST verifier
    "FORBIDDEN_INFLATE_FRAMEWORKS",
    # serialization
    "NPSD_MAGIC",
    "NPSD_SCHEMA_VERSION",
    "InflateNotNumpyPortableError",
    "NumpyPortableStateDictError",
    "assert_inflate_is_numpy_portable",
    "bilinear_resize_nhwc",
    "bilinear_upsample_2x_nhwc",
    "conv2d_nhwc",
    "conv2d_numpy",
    "film_modulate_numpy",
    "find_forbidden_framework_imports",
    "gelu",
    "gru_cell_numpy",
    "kahan_mean",
    "linear",
    "mean",
    "pack_state_dict_numpy",
    "pixel_shuffle_2x_nhwc",
    "relu",
    "sigmoid",
    "sin",
    "tanh",
    # decode primitives (re-exported canonical reference)
    "to_float32",
    "unpack_state_dict_numpy",
    # runtime emitter
    "write_numpy_portable_contest_runtime",
    "write_rgb_pair_to_raw_numpy",
]
