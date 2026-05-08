"""Factorized HNeRV runtime codec.

Implements **per-tensor SVD low-rank factorization** for the PR101/PR106/PR107
HNeRV decoder. Some tensors in the FIXED_STATE_SCHEMA carry a non-trivial
amount of redundancy; replacing the full int8 tensor with U @ diag(S) @ V^T
factor matrices (each int8-quantized with a per-component fp16 scale) and
brotli-compressing them as a joint payload yields a smaller archive when the
chosen rank ``r`` is well below ``min(M, N)`` of the unfolded tensor.

This module is the **encoder + decoder + wire-format spec** that closes
codex's ``sub017_cpu_frontier_plan_20260508_codex.md`` dispatch blocker
``factorized_hnerv_runtime_not_implemented``.

Design summary
--------------

* For a 2D weight ``W`` of shape ``(M, N)``: SVD gives ``U(M, r) @ diag(S(r,))
  @ V^T(r, N)`` for any ``r ≤ min(M, N)``. We quantize ``U``, ``S``, ``V^T``
  independently to int8 with per-array fp16 scales.
* For a 4D Conv2d weight of shape ``(O, I, kH, kW)``: we unfold along axis 0
  to ``(O, I*kH*kW)`` and factor that 2D matrix the same way, then reshape
  the reconstruction back to 4D at decode time.
* The encoder writes a small **per-tensor record** containing:
  ``rank (uint16) | scale_U (fp16) | scale_S (fp16) | scale_V (fp16) |
  U_int8_bytes | S_int8_bytes | V_int8_bytes``.
* All per-tensor records are concatenated into a single **factorized section**;
  the section is brotli-compressed at quality 11. A **section header** lists
  which schema indices were factorized and is also part of the wire format.
* The remainder of FIXED_STATE_SCHEMA (tensors not factorized) is encoded
  exactly as before — per-tensor int8 + fp16 scale, packed into a single
  brotli stream — for byte-faithful composability with the rest of the lab.

Composability
-------------

The factorized runtime is intentionally *layered*:

* **Continuous-K allocation** runs over the same schema; per-tensor K becomes
  per-component K after factorization (each of U/S/V can carry its own K).
  The default factor encoder uses int8 for all components; callers can
  substitute lossy_coarsening on the int8 streams via the helper
  :func:`apply_factor_lossy_coarsening`.
* **Analytical lossy_coarsening** treats the int8 streams as 1-D byte arrays,
  identical to PR101/PR106. The factorized streams are byte-compatible.
* **Entropy pack guards** can run on the brotli'd output. The wire format
  version-tags the section as ``factorized_hnerv_codec.v1``.

Score relevance
---------------

Per CLAUDE.md "FORBIDDEN PATTERNS": every numerical claim in this module's
docstrings is either ``[predicted]`` (uniformity-of-byte-savings projection)
or ``[empirical:<artifact>]`` (after the demo run anchors the byte map).
Strict-scorer-rule: NO scorer loads in encode or decode paths.

References
----------

* PR101 split-brotli baseline: ``tac.pr101_split_brotli_codec`` (parallel
  encoder/decoder with per-tensor byte_map permutations).
* PR106 packed-brotli baseline: ``tac.hnerv_decoder_recode``
  (``parse_packed_decoder_brotli``).
* Sub-0.17 plan: ``.omx/research/sub017_cpu_frontier_plan_20260508_codex.md``.
* Cost-curve form-uniformity contract: ``tac.codec.rel_err.RelErrForm.RMS``.
"""
from __future__ import annotations

import logging
import math
import struct
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import brotli
import numpy as np
import torch

from tac.codec.rel_err import REL_ERR_FORM_KEY, RelErrForm, compute_rel_err

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wire format identifiers
# ---------------------------------------------------------------------------

# Version tag emitted in build manifests. Section header magic on disk is
# the same string truncated to 16 bytes (right-padded with NULs).
WIRE_FORMAT_VERSION: str = "factorized_hnerv_codec.v1"

# 4-byte magic at the start of the on-disk factorized section. Distinct from
# PR101's split-brotli payload so a decoder can refuse the wrong section.
SECTION_MAGIC: bytes = b"FHN1"

# Maximum representable rank in the wire format (uint16). Practical ranks for
# our schema sit well below this (largest tensor is 1728 × 28 stem.weight).
MAX_RANK: int = 0xFFFF

# Quantization range. Identical to PR101/PR106 (signed-7-bit -> u8 zigzag).
N_QUANT: int = 127


# Schema is duplicated here to keep this module zero-dep on hnerv_decoder_recode
# (which would pull in arithmetic_qint_codec etc.). Identical to PR106's
# FIXED_STATE_SCHEMA / PR101's FIXED_STATE_SCHEMA.
FIXED_STATE_SCHEMA: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("stem.weight", (1728, 28)),
    ("stem.bias", (1728,)),
    ("blocks.0.weight", (144, 36, 3, 3)),
    ("blocks.0.bias", (144,)),
    ("blocks.1.weight", (144, 36, 3, 3)),
    ("blocks.1.bias", (144,)),
    ("blocks.2.weight", (108, 36, 3, 3)),
    ("blocks.2.bias", (108,)),
    ("blocks.3.weight", (80, 27, 3, 3)),
    ("blocks.3.bias", (80,)),
    ("blocks.4.weight", (72, 20, 3, 3)),
    ("blocks.4.bias", (72,)),
    ("blocks.5.weight", (72, 18, 3, 3)),
    ("blocks.5.bias", (72,)),
    ("skips.2.weight", (27, 36, 1, 1)),
    ("skips.2.bias", (27,)),
    ("skips.3.weight", (20, 27, 1, 1)),
    ("skips.3.bias", (20,)),
    ("skips.4.weight", (18, 20, 1, 1)),
    ("skips.4.bias", (18,)),
    ("refine.0.weight", (9, 18, 3, 3)),
    ("refine.0.bias", (9,)),
    ("refine.1.weight", (18, 9, 3, 3)),
    ("refine.1.bias", (18,)),
    ("rgb_0.weight", (3, 18, 3, 3)),
    ("rgb_0.bias", (3,)),
    ("rgb_1.weight", (3, 18, 3, 3)),
    ("rgb_1.bias", (3,)),
)

_SCHEMA_INDEX: dict[str, int] = {name: i for i, (name, _) in enumerate(FIXED_STATE_SCHEMA)}
_SCHEMA_NAMES: list[str] = [name for name, _ in FIXED_STATE_SCHEMA]


class FactorizedHnervCodecError(ValueError):
    """Raised on any factorized-codec wire-format / contract violation."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FactorizedTensor:
    """Outcome of an SVD factorization at a chosen rank.

    Components (``u``, ``s``, ``v``) are stored as int8 after symmetric
    per-component scaling; the per-component fp16 scale is the value that
    multiplies the int8 reconstruction back to the float component.

    The reconstruction equation is:

        W_reconstructed = (U_i8 * scale_U) @ diag(S_i8 * scale_S)
                           @ (V_i8 * scale_V).T            # 2D form
        # for 4D conv tensors: W_reconstructed is then reshaped to original.

    Attributes:
        name: tensor name (matches FIXED_STATE_SCHEMA).
        original_shape: full original shape (e.g. ``(144, 36, 3, 3)``).
        unfolded_shape: shape after unfolding to 2D (M, N). For 2D inputs
            equal to ``original_shape``; for 4D inputs equal to
            ``(O, I*kH*kW)``.
        rank: chosen rank ``r``; satisfies ``1 <= r <= min(M, N)``.
        u_i8: int8 array of shape ``(M, r)``.
        s_i8: int8 array of shape ``(r,)``.
        v_i8: int8 array of shape ``(r, N)`` (note: we store V^T, not V).
        scale_u: per-component fp16-castable fp32 scale for U.
        scale_s: per-component fp16-castable fp32 scale for S.
        scale_v: per-component fp16-castable fp32 scale for V^T.
        rel_err: RMS relative error of the reconstruction vs the original
            float tensor (per :class:`tac.codec.rel_err.RelErrForm.RMS`).
    """
    name: str
    original_shape: tuple[int, ...]
    unfolded_shape: tuple[int, int]
    rank: int
    u_i8: np.ndarray  # (M, r)
    s_i8: np.ndarray  # (r,)
    v_i8: np.ndarray  # (r, N)  (this is V^T, not V)
    scale_u: float
    scale_s: float
    scale_v: float
    rel_err: float

    @property
    def factor_byte_count(self) -> int:
        """Total int8 bytes in the U/S/V components (excludes scales/header)."""
        return int(self.u_i8.size + self.s_i8.size + self.v_i8.size)

    def reconstruct_float(self) -> np.ndarray:
        """Reconstruct the full float tensor from int8 components + scales."""
        u_f = self.u_i8.astype(np.float64) * float(self.scale_u)
        s_f = self.s_i8.astype(np.float64) * float(self.scale_s)
        v_f = self.v_i8.astype(np.float64) * float(self.scale_v)
        # M @ diag(S) @ N == M * S broadcast over rows of V
        m2d = (u_f * s_f[None, :]) @ v_f
        return m2d.reshape(self.original_shape).astype(np.float32)


@dataclass
class FactorizedSectionPlan:
    """Plan for which schema indices to factorize, plus per-index ranks.

    Indices not present in ``factorized_indices`` fall through to the
    non-factorized path (per-tensor int8 + fp16 scale + brotli).
    """
    factorized_indices: tuple[int, ...] = ()
    per_index_rank: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(set(self.factorized_indices)) != len(self.factorized_indices):
            raise FactorizedHnervCodecError(
                "factorized_indices must be unique; got "
                f"{self.factorized_indices}"
            )
        for idx in self.factorized_indices:
            if not 0 <= idx < len(FIXED_STATE_SCHEMA):
                raise FactorizedHnervCodecError(
                    f"factorized_indices entry {idx} out of range"
                )
            if idx not in self.per_index_rank:
                raise FactorizedHnervCodecError(
                    f"per_index_rank missing rank for factorized index {idx}"
                )
            r = self.per_index_rank[idx]
            if not 1 <= r <= MAX_RANK:
                raise FactorizedHnervCodecError(
                    f"per_index_rank[{idx}]={r} out of allowed range "
                    f"[1, {MAX_RANK}]"
                )


# ---------------------------------------------------------------------------
# Quantization helper
# ---------------------------------------------------------------------------

def _quantize_array(arr: np.ndarray, n_quant: int = N_QUANT) -> tuple[np.ndarray, float]:
    """Symmetric per-array INT8 quantization. Returns (i8_array, scale)."""
    arr_f = arr.astype(np.float64)
    abs_max = float(np.abs(arr_f).max()) if arr_f.size else 0.0
    scale = abs_max / n_quant if abs_max > 0 else 1.0
    q = np.round(arr_f / scale).clip(-n_quant, n_quant).astype(np.int8)
    # Round-trip the scale through fp16 so quantization error matches the
    # actual on-disk reconstruction.
    scale_f16 = float(np.float16(scale))
    if scale_f16 == 0.0:  # extreme tensor with abs_max = 0
        scale_f16 = 1.0
    return q, scale_f16


# ---------------------------------------------------------------------------
# SVD factorization
# ---------------------------------------------------------------------------

def _unfold_tensor(t: torch.Tensor) -> tuple[np.ndarray, tuple[int, ...], tuple[int, int]]:
    """Unfold a torch tensor to a 2D matrix for SVD.

    For 1D inputs (biases) we refuse: those should not be factorized — the
    encoder caller is expected to leave biases out of the factorized index
    set.
    """
    arr = t.detach().cpu().numpy().astype(np.float64)
    original_shape = tuple(int(s) for s in t.shape)
    if arr.ndim == 1:
        raise FactorizedHnervCodecError(
            "SVD factorization of 1D tensor (bias) is not supported; "
            "exclude this index from factorized_indices."
        )
    if arr.ndim == 2:
        return arr, original_shape, (arr.shape[0], arr.shape[1])
    if arr.ndim == 4:
        # (O, I, kH, kW) -> (O, I*kH*kW)
        m = arr.shape[0]
        n = int(arr.shape[1] * arr.shape[2] * arr.shape[3])
        return arr.reshape(m, n), original_shape, (m, n)
    raise FactorizedHnervCodecError(
        f"unsupported tensor ndim={arr.ndim} for SVD factorization"
    )


def factorize_tensor_svd(
    name: str,
    tensor: torch.Tensor,
    *,
    target_rank: int | None = None,
    target_rms_err: float | None = None,
    n_quant: int = N_QUANT,
) -> FactorizedTensor:
    """SVD-factorize a 2D / 4D tensor into int8-quantized U, S, V^T components.

    Either ``target_rank`` OR ``target_rms_err`` (or both) must be supplied.
    When both are given we use ``target_rank`` and report the achieved RMS
    error in the returned record. When only ``target_rms_err`` is given we
    bisect over rank to find the smallest ``r`` whose post-quantization
    reconstruction has RMS rel_err ``<= target_rms_err``.

    Args:
        name: tensor name (kept on the record for round-trip validation).
        tensor: the float tensor to factorize.
        target_rank: explicit rank ``r`` (1..min(M, N)).
        target_rms_err: target RMS relative error
            (:class:`tac.codec.rel_err.RelErrForm.RMS`). When supplied without
            ``target_rank`` we bisect over the available ranks; when both are
            supplied we use the explicit ``target_rank``.
        n_quant: int8 quantization range (default 127).

    Returns:
        FactorizedTensor with int8 U/S/V^T, per-component fp16 scales, and
        the resulting reconstruction RMS rel_err.

    Raises:
        FactorizedHnervCodecError: on shape / parameter violations.
    """
    if target_rank is None and target_rms_err is None:
        raise FactorizedHnervCodecError(
            "factorize_tensor_svd requires target_rank or target_rms_err"
        )

    unfolded, original_shape, (M, N) = _unfold_tensor(tensor)
    max_rank = min(M, N)
    if max_rank < 1:
        raise FactorizedHnervCodecError(
            f"tensor {name!r} has degenerate unfolded shape {(M, N)}"
        )

    # SVD once at full rank; we slice the first ``r`` components below.
    # ``np.linalg.svd`` with full_matrices=False returns
    # u (M, k), s (k,), vt (k, N) where k = min(M, N).
    u_full, s_full, vt_full = np.linalg.svd(unfolded, full_matrices=False)

    def _build_at_rank(r: int) -> FactorizedTensor:
        u_r = u_full[:, :r].astype(np.float64)
        s_r = s_full[:r].astype(np.float64)
        v_r = vt_full[:r, :].astype(np.float64)
        u_i8, scale_u = _quantize_array(u_r, n_quant=n_quant)
        s_i8, scale_s = _quantize_array(s_r, n_quant=n_quant)
        v_i8, scale_v = _quantize_array(v_r, n_quant=n_quant)
        # Reconstruction with quantized components
        u_d = u_i8.astype(np.float64) * scale_u
        s_d = s_i8.astype(np.float64) * scale_s
        v_d = v_i8.astype(np.float64) * scale_v
        m2d = (u_d * s_d[None, :]) @ v_d
        rel = compute_rel_err(m2d, unfolded, mode=RelErrForm.RMS)
        return FactorizedTensor(
            name=name,
            original_shape=original_shape,
            unfolded_shape=(M, N),
            rank=r,
            u_i8=u_i8,
            s_i8=s_i8,
            v_i8=v_i8,
            scale_u=scale_u,
            scale_s=scale_s,
            scale_v=scale_v,
            rel_err=rel,
        )

    if target_rank is not None:
        r = max(1, min(int(target_rank), max_rank))
        return _build_at_rank(r)

    # bisect over rank for target_rms_err
    assert target_rms_err is not None
    # try smallest rank that satisfies; fall back to max_rank
    lo, hi = 1, max_rank
    best: FactorizedTensor | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = _build_at_rank(mid)
        if cand.rel_err <= target_rms_err:
            best = cand
            hi = mid - 1
        else:
            lo = mid + 1
    if best is None:
        # Even at max_rank we can't hit the target; return max_rank.
        best = _build_at_rank(max_rank)
    return best


# ---------------------------------------------------------------------------
# Wire format encode / decode
# ---------------------------------------------------------------------------

# Per-tensor record on disk:
#   uint16 schema_idx
#   uint16 rank
#   uint8  ndim_original  (2 or 4)
#   uint16 shape_dim0
#   uint16 shape_dim1
#   uint16 shape_dim2  (0 if 2D)
#   uint16 shape_dim3  (0 if 2D)
#   fp16   scale_u
#   fp16   scale_s
#   fp16   scale_v
#   int8 * (M*r) U bytes
#   int8 * r     S bytes
#   int8 * (r*N) V bytes  (this is V^T)
#
# All ints are little-endian.

_PER_TENSOR_HEADER = struct.Struct(
    "<HH"   # schema_idx, rank
    "B"     # ndim_original
    "HHHH"  # shape (4 dims, zero-padded)
    "eee"   # scale_u, scale_s, scale_v (fp16)
)


def _encode_one_factorized_record(ft: FactorizedTensor) -> bytes:
    """Serialize one FactorizedTensor to bytes."""
    if ft.name not in _SCHEMA_INDEX:
        raise FactorizedHnervCodecError(
            f"unknown tensor name {ft.name!r}; not in FIXED_STATE_SCHEMA"
        )
    idx = _SCHEMA_INDEX[ft.name]
    M, N = ft.unfolded_shape
    if ft.u_i8.shape != (M, ft.rank):
        raise FactorizedHnervCodecError(
            f"u_i8 shape {ft.u_i8.shape} != ({M}, {ft.rank}) for {ft.name!r}"
        )
    if ft.s_i8.shape != (ft.rank,):
        raise FactorizedHnervCodecError(
            f"s_i8 shape {ft.s_i8.shape} != ({ft.rank},) for {ft.name!r}"
        )
    if ft.v_i8.shape != (ft.rank, N):
        raise FactorizedHnervCodecError(
            f"v_i8 shape {ft.v_i8.shape} != ({ft.rank}, {N}) for {ft.name!r}"
        )
    if not (1 <= ft.rank <= MAX_RANK):
        raise FactorizedHnervCodecError(f"rank {ft.rank} out of range")

    ndim = len(ft.original_shape)
    if ndim not in (2, 4):
        raise FactorizedHnervCodecError(
            f"unsupported ndim {ndim} for tensor {ft.name!r}"
        )
    s0 = int(ft.original_shape[0])
    s1 = int(ft.original_shape[1])
    s2 = int(ft.original_shape[2]) if ndim == 4 else 0
    s3 = int(ft.original_shape[3]) if ndim == 4 else 0

    header = _PER_TENSOR_HEADER.pack(
        idx, ft.rank, ndim, s0, s1, s2, s3,
        np.float16(ft.scale_u), np.float16(ft.scale_s), np.float16(ft.scale_v),
    )
    return (
        header
        + ft.u_i8.tobytes()
        + ft.s_i8.tobytes()
        + ft.v_i8.tobytes()
    )


def _decode_one_factorized_record(buf: memoryview, pos: int) -> tuple[FactorizedTensor, int]:
    """Read one record from ``buf`` starting at ``pos``; return (record, new_pos)."""
    header_size = _PER_TENSOR_HEADER.size
    if pos + header_size > len(buf):
        raise FactorizedHnervCodecError(
            "truncated factorized record (header overflow)"
        )
    fields = _PER_TENSOR_HEADER.unpack_from(buf, pos)
    idx, rank, ndim, s0, s1, s2, s3, sU, sS, sV = fields
    pos += header_size
    if not 0 <= idx < len(FIXED_STATE_SCHEMA):
        raise FactorizedHnervCodecError(f"factorized record idx={idx} out of range")
    if not 1 <= rank <= MAX_RANK:
        raise FactorizedHnervCodecError(f"factorized record rank={rank} out of range")
    if ndim not in (2, 4):
        raise FactorizedHnervCodecError(f"factorized record ndim={ndim} unsupported")

    schema_name, schema_shape = FIXED_STATE_SCHEMA[idx]
    if ndim == 2:
        original_shape = (s0, s1)
        if original_shape != schema_shape:
            raise FactorizedHnervCodecError(
                f"shape mismatch idx={idx}: schema {schema_shape}, on-disk {original_shape}"
            )
        M, N = s0, s1
    else:
        original_shape = (s0, s1, s2, s3)
        if original_shape != schema_shape:
            raise FactorizedHnervCodecError(
                f"shape mismatch idx={idx}: schema {schema_shape}, on-disk {original_shape}"
            )
        M = s0
        N = int(s1 * s2 * s3)

    u_size = M * rank
    s_size = rank
    v_size = rank * N
    end = pos + u_size + s_size + v_size
    if end > len(buf):
        raise FactorizedHnervCodecError(
            f"truncated factorized record (idx={idx}, rank={rank}, "
            f"need {u_size + s_size + v_size} bytes, have {len(buf) - pos})"
        )
    u_bytes = bytes(buf[pos:pos + u_size]); pos += u_size
    s_bytes = bytes(buf[pos:pos + s_size]); pos += s_size
    v_bytes = bytes(buf[pos:pos + v_size]); pos += v_size

    u_i8 = np.frombuffer(u_bytes, dtype=np.int8).reshape(M, rank).copy()
    s_i8 = np.frombuffer(s_bytes, dtype=np.int8).reshape(rank).copy()
    v_i8 = np.frombuffer(v_bytes, dtype=np.int8).reshape(rank, N).copy()

    return (
        FactorizedTensor(
            name=schema_name,
            original_shape=original_shape,
            unfolded_shape=(M, N),
            rank=int(rank),
            u_i8=u_i8,
            s_i8=s_i8,
            v_i8=v_i8,
            scale_u=float(sU),
            scale_s=float(sS),
            scale_v=float(sV),
            rel_err=float("nan"),  # not stored on disk; recomputable if caller needs it
        ),
        pos,
    )


# ---------------------------------------------------------------------------
# Non-factorized helper (pass-through int8 packing)
# ---------------------------------------------------------------------------

def _encode_non_factorized_tensor(name: str, tensor: torch.Tensor) -> bytes:
    """Per-tensor INT8 + fp16 scale, raw bytes (no brotli wrapping yet).

    Wire layout (one tensor):
        uint16 schema_idx
        fp16   scale
        int8 * prod(shape) values
    """
    if name not in _SCHEMA_INDEX:
        raise FactorizedHnervCodecError(f"unknown tensor name {name!r}")
    idx = _SCHEMA_INDEX[name]
    schema_shape = FIXED_STATE_SCHEMA[idx][1]
    if tuple(int(s) for s in tensor.shape) != schema_shape:
        raise FactorizedHnervCodecError(
            f"shape mismatch for {name!r}: tensor={tuple(tensor.shape)}, "
            f"schema={schema_shape}"
        )
    arr = tensor.detach().cpu().float().numpy().astype(np.float64)
    abs_max = float(np.abs(arr).max()) if arr.size else 0.0
    scale = abs_max / N_QUANT if abs_max > 0 else 1.0
    scale = float(np.float16(scale)) or 1.0
    q = np.round(arr / scale).clip(-N_QUANT, N_QUANT).astype(np.int8)
    return (
        struct.pack("<H", idx)
        + np.float16(scale).tobytes()
        + q.tobytes()
    )


def _decode_one_non_factorized(buf: memoryview, pos: int) -> tuple[str, np.ndarray, float, int]:
    """Read one non-factorized record; return (name, q_i8, scale_fp32, new_pos)."""
    if pos + 2 + 2 > len(buf):
        raise FactorizedHnervCodecError(
            "truncated non-factorized record (header overflow)"
        )
    (idx,) = struct.unpack_from("<H", buf, pos); pos += 2
    if not 0 <= idx < len(FIXED_STATE_SCHEMA):
        raise FactorizedHnervCodecError(
            f"non-factorized record idx={idx} out of range"
        )
    name, shape = FIXED_STATE_SCHEMA[idx]
    scale = float(np.frombuffer(bytes(buf[pos:pos + 2]), dtype=np.float16)[0]); pos += 2
    n = int(np.prod(shape))
    if pos + n > len(buf):
        raise FactorizedHnervCodecError(
            f"truncated non-factorized record (idx={idx}, need {n} bytes)"
        )
    q = np.frombuffer(bytes(buf[pos:pos + n]), dtype=np.int8).reshape(shape).copy()
    pos += n
    return name, q, scale, pos


# ---------------------------------------------------------------------------
# Top-level encode / decode
# ---------------------------------------------------------------------------

# Section header on disk:
#   4 bytes magic "FHN1"
#   uint16 num_factorized
#   uint16 num_non_factorized
#   uint32 factorized_payload_len  (post-brotli compressed length)
#   uint32 non_factorized_payload_len  (post-brotli compressed length)

_SECTION_HEADER = struct.Struct("<4sHHII")


def encode_factorized_section(
    state_dict: dict[str, torch.Tensor],
    plan: FactorizedSectionPlan,
    *,
    target_rms_err_per_tensor: dict[int, float] | None = None,
    brotli_quality: int = 11,
) -> tuple[bytes, dict[str, object]]:
    """Encode a state_dict into a factorized-HNeRV section.

    Args:
        state_dict: torch state dict keyed by FIXED_STATE_SCHEMA names. Must
            contain every name from the schema (factorized or not).
        plan: which schema indices to factorize and at what rank.
        target_rms_err_per_tensor: optional per-index RMS-err target. When
            provided AND the index is in ``plan.factorized_indices``, we
            bisect rank starting from ``plan.per_index_rank[idx]`` only if
            the requested rank doesn't already meet the target. We always
            honor ``plan.per_index_rank[idx]`` first; ``target_rms_err`` is
            only used to validate the chosen rank and emit a warning if the
            achieved rel_err exceeds the target.
        brotli_quality: brotli quality for the two compressed streams.

    Returns:
        Tuple ``(section_bytes, telemetry)`` where ``telemetry`` is a dict
        with per-tensor ranks, rel_errs, factor byte counts, and the post-
        brotli stream lengths.
    """
    # 1. Validate state_dict completeness.
    missing = [n for n in _SCHEMA_NAMES if n not in state_dict]
    if missing:
        raise FactorizedHnervCodecError(
            f"state_dict missing tensors: {missing[:3]}{'...' if len(missing) > 3 else ''}"
        )

    # 2. Factorize all indices in the plan.
    factorized_records: list[FactorizedTensor] = []
    per_index_rel_err: dict[int, float] = {}
    for idx in plan.factorized_indices:
        name, _shape = FIXED_STATE_SCHEMA[idx]
        ft = factorize_tensor_svd(
            name=name,
            tensor=state_dict[name],
            target_rank=plan.per_index_rank[idx],
        )
        target_rms = (
            target_rms_err_per_tensor.get(idx) if target_rms_err_per_tensor else None
        )
        if target_rms is not None and ft.rel_err > target_rms:
            logger.warning(
                "factorized tensor %r at rank=%d has rel_err=%.4f > target %.4f",
                name, ft.rank, ft.rel_err, target_rms,
            )
        per_index_rel_err[idx] = ft.rel_err
        factorized_records.append(ft)

    # 3. Encode non-factorized indices in schema order.
    non_factorized_indices = tuple(
        i for i in range(len(FIXED_STATE_SCHEMA)) if i not in plan.factorized_indices
    )
    non_factorized_payload_parts: list[bytes] = []
    for idx in non_factorized_indices:
        name, _shape = FIXED_STATE_SCHEMA[idx]
        non_factorized_payload_parts.append(
            _encode_non_factorized_tensor(name, state_dict[name])
        )
    non_factorized_raw = b"".join(non_factorized_payload_parts)

    # 4. Encode factorized records (concatenated) and brotli them.
    factorized_raw = b"".join(_encode_one_factorized_record(ft) for ft in factorized_records)

    factorized_brotli = brotli.compress(factorized_raw, quality=brotli_quality) if factorized_raw else b""
    non_factorized_brotli = brotli.compress(non_factorized_raw, quality=brotli_quality) if non_factorized_raw else b""

    # 5. Build section header + payloads.
    if len(plan.factorized_indices) > 0xFFFF or len(non_factorized_indices) > 0xFFFF:
        raise FactorizedHnervCodecError("too many tensors for uint16 counts in header")

    header = _SECTION_HEADER.pack(
        SECTION_MAGIC,
        len(plan.factorized_indices),
        len(non_factorized_indices),
        len(factorized_brotli),
        len(non_factorized_brotli),
    )
    # Index lists follow the header so the decoder knows what's there.
    idx_table = (
        np.array(plan.factorized_indices, dtype=np.uint16).tobytes()
        + np.array(non_factorized_indices, dtype=np.uint16).tobytes()
    )

    section = header + idx_table + factorized_brotli + non_factorized_brotli

    telemetry = {
        "wire_format_version": WIRE_FORMAT_VERSION,
        "factorized_indices": list(plan.factorized_indices),
        "non_factorized_indices": list(non_factorized_indices),
        "per_tensor_ranks": [int(ft.rank) for ft in factorized_records],
        "per_tensor_rel_errs": [float(ft.rel_err) for ft in factorized_records],
        "per_tensor_rel_err_form": RelErrForm.RMS.value,
        "factor_section_bytes": len(factorized_brotli),
        "non_factorized_section_bytes": len(non_factorized_brotli),
        "section_total_bytes": len(section),
        "factorized_raw_bytes": len(factorized_raw),
        "non_factorized_raw_bytes": len(non_factorized_raw),
        REL_ERR_FORM_KEY: RelErrForm.RMS.value,
    }
    return section, telemetry


def decode_factorized_section(data: bytes) -> dict[str, torch.Tensor]:
    """Decode a factorized-HNeRV section back into a torch state_dict.

    Returns a state_dict containing every tensor in FIXED_STATE_SCHEMA, in
    schema order.

    Raises:
        FactorizedHnervCodecError: on any wire-format violation.
    """
    if len(data) < _SECTION_HEADER.size:
        raise FactorizedHnervCodecError("data shorter than section header")
    magic, n_fact, n_non_fact, fact_len, non_fact_len = _SECTION_HEADER.unpack_from(data, 0)
    if magic != SECTION_MAGIC:
        raise FactorizedHnervCodecError(
            f"bad section magic: expected {SECTION_MAGIC!r}, got {magic!r}"
        )
    pos = _SECTION_HEADER.size

    expected_idx_table_bytes = (n_fact + n_non_fact) * 2
    if pos + expected_idx_table_bytes > len(data):
        raise FactorizedHnervCodecError("truncated index table")
    idx_table = np.frombuffer(
        data, dtype=np.uint16, count=(n_fact + n_non_fact), offset=pos
    )
    pos += expected_idx_table_bytes
    factorized_indices = tuple(int(i) for i in idx_table[:n_fact])
    non_factorized_indices = tuple(int(i) for i in idx_table[n_fact:])

    if pos + fact_len + non_fact_len != len(data):
        raise FactorizedHnervCodecError(
            f"section length mismatch: expected {pos + fact_len + non_fact_len}, got {len(data)}"
        )
    factorized_brotli = data[pos:pos + fact_len]; pos += fact_len
    non_factorized_brotli = data[pos:pos + non_fact_len]; pos += non_fact_len

    # Decompress and parse factorized records
    factorized_raw = brotli.decompress(factorized_brotli) if factorized_brotli else b""
    fact_buf = memoryview(factorized_raw)
    fact_pos = 0
    parsed_factorized: dict[int, FactorizedTensor] = {}
    for _ in range(n_fact):
        ft, fact_pos = _decode_one_factorized_record(fact_buf, fact_pos)
        idx = _SCHEMA_INDEX[ft.name]
        parsed_factorized[idx] = ft
    if fact_pos != len(factorized_raw):
        raise FactorizedHnervCodecError(
            "trailing bytes in factorized payload"
        )
    if set(parsed_factorized.keys()) != set(factorized_indices):
        raise FactorizedHnervCodecError(
            f"factorized index table {factorized_indices} disagrees with payload "
            f"{sorted(parsed_factorized.keys())}"
        )

    # Decompress and parse non-factorized records
    non_factorized_raw = brotli.decompress(non_factorized_brotli) if non_factorized_brotli else b""
    nf_buf = memoryview(non_factorized_raw)
    nf_pos = 0
    parsed_non_factorized: dict[int, tuple[np.ndarray, float]] = {}
    for _ in range(n_non_fact):
        name, q, scale, nf_pos = _decode_one_non_factorized(nf_buf, nf_pos)
        idx = _SCHEMA_INDEX[name]
        parsed_non_factorized[idx] = (q, scale)
    if nf_pos != len(non_factorized_raw):
        raise FactorizedHnervCodecError(
            "trailing bytes in non-factorized payload"
        )
    if set(parsed_non_factorized.keys()) != set(non_factorized_indices):
        raise FactorizedHnervCodecError(
            f"non-factorized index table {non_factorized_indices} disagrees with "
            f"payload {sorted(parsed_non_factorized.keys())}"
        )

    # Build state_dict in schema order
    sd: dict[str, torch.Tensor] = {}
    for idx, (name, _shape) in enumerate(FIXED_STATE_SCHEMA):
        if idx in parsed_factorized:
            ft = parsed_factorized[idx]
            recon = ft.reconstruct_float()
            sd[name] = torch.from_numpy(recon.astype(np.float32))
        elif idx in parsed_non_factorized:
            q, scale = parsed_non_factorized[idx]
            sd[name] = torch.from_numpy(q.astype(np.float32)) * float(scale)
        else:
            raise FactorizedHnervCodecError(
                f"schema idx {idx} ({_SCHEMA_NAMES[idx]}) not in either index table"
            )
    return sd


# ---------------------------------------------------------------------------
# Cross-paradigm composability hooks
# ---------------------------------------------------------------------------

def apply_factor_lossy_coarsening(
    ft: FactorizedTensor,
    *,
    K_u: int = 1,
    K_s: int = 1,
    K_v: int = 1,
) -> FactorizedTensor:
    """Apply analytical lossy_coarsening to U/S/V int8 streams independently.

    Coarsening with parameter K replaces every value v with
    ``round(v / K) * K``, reducing the effective alphabet of the int8 stream.
    K=1 is a no-op. Recommend K_s = 1 (singular values are already low-rank-
    sparse) and small K_u / K_v for the dense factor matrices.

    Returns a new FactorizedTensor with coarsened components and a re-computed
    ``rel_err`` (relative to the un-factorized target stored on the input
    record's reconstruction; not the original tensor — caller must recompute
    that themselves if they need it).
    """
    if K_u < 1 or K_s < 1 or K_v < 1:
        raise FactorizedHnervCodecError("lossy K parameters must be >= 1")

    def coarsen(arr: np.ndarray, K: int) -> np.ndarray:
        if K == 1:
            return arr
        return (np.round(arr.astype(np.int32) / K) * K).clip(-N_QUANT, N_QUANT).astype(np.int8)

    new_u = coarsen(ft.u_i8, K_u)
    new_s = coarsen(ft.s_i8, K_s)
    new_v = coarsen(ft.v_i8, K_v)

    # Recompute rel_err vs the input's reconstruction (the caller should
    # re-compute vs the original tensor at a higher level if needed).
    orig_recon = ft.reconstruct_float()
    new_record = FactorizedTensor(
        name=ft.name,
        original_shape=ft.original_shape,
        unfolded_shape=ft.unfolded_shape,
        rank=ft.rank,
        u_i8=new_u,
        s_i8=new_s,
        v_i8=new_v,
        scale_u=ft.scale_u,
        scale_s=ft.scale_s,
        scale_v=ft.scale_v,
        rel_err=float("nan"),
    )
    new_recon = new_record.reconstruct_float()
    rel = compute_rel_err(new_recon, orig_recon, mode=RelErrForm.RMS)
    return FactorizedTensor(
        name=ft.name,
        original_shape=ft.original_shape,
        unfolded_shape=ft.unfolded_shape,
        rank=ft.rank,
        u_i8=new_u,
        s_i8=new_s,
        v_i8=new_v,
        scale_u=ft.scale_u,
        scale_s=ft.scale_s,
        scale_v=ft.scale_v,
        rel_err=rel,
    )


# ---------------------------------------------------------------------------
# Convenience API for build-time planners
# ---------------------------------------------------------------------------

def estimate_factorized_byte_savings(
    state_dict: dict[str, torch.Tensor],
    candidate_indices: Iterable[int],
    *,
    target_ranks: dict[int, int] | None = None,
    target_rms_err: float | None = None,
    brotli_quality: int = 11,
) -> dict[int, dict[str, object]]:
    """Per-candidate-index byte-savings estimate.

    For each candidate index, factorize at the chosen rank (or bisect for
    target_rms_err) and compare the brotli-compressed factorized record vs
    the brotli-compressed non-factorized record on its own. Returns a dict
    keyed by index with raw + brotli'd bytes for both representations and
    the achieved rel_err. Note: brotli context-sharing across the section
    means the actual section savings will differ from the per-tensor sum.

    [predicted]: this is a per-tensor isolated estimate; cross-tensor brotli
    redundancy is NOT modeled here. Use :func:`encode_factorized_section`
    for the empirical section bytes.
    """
    out: dict[int, dict[str, object]] = {}
    for idx in candidate_indices:
        name, _shape = FIXED_STATE_SCHEMA[idx]
        if name not in state_dict:
            raise FactorizedHnervCodecError(f"state_dict missing {name!r}")

        target_rank = (target_ranks or {}).get(idx)
        ft = factorize_tensor_svd(
            name=name, tensor=state_dict[name],
            target_rank=target_rank, target_rms_err=target_rms_err,
        )
        fact_record = _encode_one_factorized_record(ft)
        non_fact_record = _encode_non_factorized_tensor(name, state_dict[name])

        # Apply brotli per-record for an isolated estimate (warning: cross-
        # tensor context isn't modeled).
        fact_brotli = brotli.compress(fact_record, quality=brotli_quality)
        non_fact_brotli = brotli.compress(non_fact_record, quality=brotli_quality)

        out[idx] = {
            "name": name,
            "rank": int(ft.rank),
            "rel_err": float(ft.rel_err),
            REL_ERR_FORM_KEY: RelErrForm.RMS.value,
            "factor_record_raw_bytes": len(fact_record),
            "non_factor_record_raw_bytes": len(non_fact_record),
            "factor_record_brotli_bytes": len(fact_brotli),
            "non_factor_record_brotli_bytes": len(non_fact_brotli),
            "isolated_savings_bytes_raw": len(non_fact_record) - len(fact_record),
            "isolated_savings_bytes_brotli": len(non_fact_brotli) - len(fact_brotli),
        }
    return out


__all__ = [
    "FIXED_STATE_SCHEMA",
    "FactorizedHnervCodecError",
    "FactorizedSectionPlan",
    "FactorizedTensor",
    "MAX_RANK",
    "N_QUANT",
    "SECTION_MAGIC",
    "WIRE_FORMAT_VERSION",
    "apply_factor_lossy_coarsening",
    "decode_factorized_section",
    "encode_factorized_section",
    "estimate_factorized_byte_savings",
    "factorize_tensor_svd",
]
