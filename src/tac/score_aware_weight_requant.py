"""Score-aware per-tensor weight re-quantization (Task #69).

THE HYPOTHESIS (audit #1 lurking exact-score mover)
----------------------------------------------------
The frontier HNeRV decoder weights are quantized for **pixel-recon fidelity**,
not for the minimal bytes that hold the **scored** quantities (SegNet argmax
d_seg + 6 PoseNet dims d_pose). The frozen contest scorer tolerates far more
weight error than pixel recon does. So **per-tensor bit RE-ALLOCATION by
measured scorer sensitivity** can crush the low-sensitivity tensors (lower
effective bit-depth / coarser q-grid → lower q-byte entropy → fewer entropy-
coded bytes) while holding d_seg / d_pose inside the same evaluator cell.

This module is **lossy-on-pixels, (near-)lossless-on-SCORE**. The decoder grammar
(PR #101 per-tensor int8 q-bytes + fp16 scale, byte-maps, ctx range coder) is
preserved exactly; the only change is the **effective number of quantization
levels per tensor**, chosen by the per-tensor scorer-sensitivity ranking.

NO-FAKE discipline (CLAUDE.md):
  * The re-quant ACTUALLY changes the stored q-bytes (class 1 — not a no-op).
  * Sensitivity is measured on the EXACT frozen scorer / GT decode path by the
    caller (class 8 — exact authority, not a weight-domain proxy). This module
    provides the q-domain transform + byte/entropy accounting only; the score
    measurement lives in the tool harness that decodes + runs SegNet/PoseNet.
  * The bit-allocation METHOD (rank tensors by measured frozen-scorer
    sensitivity, then waterfill bits) is ORIGINAL (class 7) — no competitor
    allocates decoder bits by per-tensor frozen-scorer sensitivity.

The q-domain transform here is intentionally pure-numpy + portable (no torch
required) so it doubles as the deterministic reference for any native port.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "requant_signed_q",
    "decode_byte_map_u8",
    "encode_byte_map_u8",
    "byte_map_roundtrip_is_identity",
    "q_byte_entropy_bits",
    "effective_levels_for_bits",
    "TensorRequantPlan",
    "allocate_bits_by_sensitivity",
    "contest_score_from_components",
    "score_delta_components",
]


# ---------------------------------------------------------------------------
# Byte-map encode/decode (inverse pair of codec.decode_mapped_u8)
# ---------------------------------------------------------------------------
# The PR #101 grammar stores each tensor's signed int8 q-values through one of
# four reversible byte maps so the resulting uint8 stream is entropy-friendly.
# decode_mapped_u8 (in the inflate-side codec) maps uint8 -> signed int8; we
# need the EXACT inverse to re-pack re-quantized q-values back into the grammar.

_VALID_BYTE_MAPS = ("zig", "negzig", "twos", "off")


def decode_byte_map_u8(arr_u8: np.ndarray, byte_map: str) -> np.ndarray:
    """uint8 stored stream -> signed int8 q-values (mirror of codec.decode_mapped_u8)."""
    arr_u8 = np.asarray(arr_u8, dtype=np.uint8)
    if byte_map == "zig":
        arr = arr_u8.astype(np.int32)
        return np.where(arr % 2 == 0, arr // 2, -(arr // 2) - 1).astype(np.int8)
    if byte_map == "negzig":
        arr = arr_u8.astype(np.int32)
        zz = np.where(arr % 2 == 0, arr // 2, -(arr // 2) - 1).astype(np.int16)
        return (-zz).astype(np.int8)
    if byte_map == "off":
        return (arr_u8.astype(np.int16) - 128).astype(np.int8)
    if byte_map == "twos":
        return arr_u8.view(np.int8)
    raise ValueError(f"unknown decoder byte map: {byte_map!r}")


def encode_byte_map_u8(q_signed: np.ndarray, byte_map: str) -> np.ndarray:
    """Signed int8 q-values -> uint8 stored stream (exact inverse of decode_byte_map_u8)."""
    q = np.asarray(q_signed).astype(np.int16)
    if byte_map == "twos":
        return (q & 0xFF).astype(np.uint8)
    if byte_map == "off":
        return (q + 128).astype(np.uint8)
    if byte_map == "zig":
        return np.where(q >= 0, 2 * q, -2 * q - 1).astype(np.uint8)
    if byte_map == "negzig":
        qn = -q
        return np.where(qn >= 0, 2 * qn, -2 * qn - 1).astype(np.uint8)
    raise ValueError(f"unknown decoder byte map: {byte_map!r}")


def byte_map_roundtrip_is_identity(byte_map: str, *, n: int = 256) -> bool:
    """True iff encode∘decode is the identity over the full int8 range for byte_map."""
    u8 = np.arange(min(n, 256), dtype=np.uint8)
    signed = decode_byte_map_u8(u8, byte_map)
    back = encode_byte_map_u8(signed, byte_map)
    return bool(np.array_equal(back, u8))


# ---------------------------------------------------------------------------
# Re-quantization in the signed q-domain
# ---------------------------------------------------------------------------


def effective_levels_for_bits(bits: float) -> int:
    """Number of representable levels for a given effective bit-depth.

    bits=8 -> 256 (no-op, full int8), bits=4 -> 16, bits=3 -> 8, bits=2 -> 4,
    bits=1 -> 2. Fractional bits are allowed (rounded levels) for fine sweeps.
    """
    if bits >= 8.0:
        return 256
    if bits <= 0.0:
        raise ValueError("bits must be > 0")
    return max(2, int(round(2.0**bits)))


def requant_signed_q(q_signed: np.ndarray, levels: int) -> np.ndarray:
    """Re-quantize signed int8 q-values onto a coarser ``levels``-step grid.

    The grid spans the full stored int8 range [-127, 127] with ``levels``
    equally-spaced reconstruction points. Keeping the per-tensor fp16 scale
    fixed, this lowers the *effective bit-depth* of the tensor: the q-byte
    stream collapses onto a small support set (lower entropy → fewer
    entropy-coded bytes) while the dequantized weight stays within one
    coarse-grid step of the original (bounded, controlled pixel error).

    levels=256 is the identity (full int8). The transform is deterministic
    and pure-numpy (portable reference).
    """
    q = np.asarray(q_signed).astype(np.int16)
    if levels >= 256:
        return q.astype(np.int8)
    if levels < 2:
        raise ValueError("levels must be >= 2")
    step = 255.0 / (levels - 1)
    rq = np.round(q.astype(np.float64) / step) * step
    return np.clip(rq, -127, 127).astype(np.int8)


def q_byte_entropy_bits(arr_u8: np.ndarray) -> float:
    """Shannon entropy (bits/symbol) of a uint8 stream — the ctx-coder rate proxy."""
    arr_u8 = np.asarray(arr_u8, dtype=np.uint8)
    if arr_u8.size == 0:
        return 0.0
    counts = np.bincount(arr_u8, minlength=256).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# ---------------------------------------------------------------------------
# Per-tensor plan + sensitivity-driven bit allocation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorRequantPlan:
    """Per-tensor re-quant assignment: storage_index -> target levels."""

    storage_index: int
    name: str
    numel: int
    levels: int

    @property
    def effective_bits(self) -> float:
        return float(np.log2(self.levels))


def allocate_bits_by_sensitivity(
    *,
    sensitivities: Mapping[int, float],
    numels: Mapping[int, int],
    names: Mapping[int, str],
    levels_ladder: Sequence[int] = (256, 64, 32, 16, 8, 4, 2),
    sensitivity_threshold: float,
    protect_top_k: int = 0,
) -> dict[int, TensorRequantPlan]:
    """Map measured per-tensor scorer sensitivity -> a re-quant level per tensor.

    The allocation is the ORIGINAL score-aware bit-allocator: tensors are ranked
    by measured frozen-scorer sensitivity (Δscore caused by a fixed q-grid
    coarsening probe). High-sensitivity tensors keep full precision (levels=256);
    low-sensitivity tensors are pushed down the ``levels_ladder`` proportionally
    to how far below ``sensitivity_threshold`` they sit. ``protect_top_k`` always
    keeps the top-k most sensitive tensors at full precision regardless.

    Returns storage_index -> TensorRequantPlan.
    """
    if not sensitivities:
        raise ValueError("sensitivities is empty")
    order = sorted(sensitivities, key=lambda k: sensitivities[k], reverse=True)
    protected = set(order[:max(0, protect_top_k)])

    smax = max(sensitivities.values())
    smax = smax if smax > 0 else 1.0

    plans: dict[int, TensorRequantPlan] = {}
    n_steps = len(levels_ladder)
    for sidx in sensitivities:
        s = sensitivities[sidx]
        if sidx in protected or s >= sensitivity_threshold:
            levels = levels_ladder[0]  # full precision
        else:
            # Below threshold: deeper crush the lower the (normalized) sensitivity.
            # rel in [0,1]: 0 == at threshold (mild crush), 1 == zero sensitivity
            # (max crush).
            rel = 1.0 - (s / sensitivity_threshold if sensitivity_threshold > 0 else 0.0)
            rel = float(np.clip(rel, 0.0, 1.0))
            step_idx = 1 + int(round(rel * (n_steps - 2)))  # skip ladder[0]=full
            step_idx = int(np.clip(step_idx, 1, n_steps - 1))
            levels = levels_ladder[step_idx]
        plans[sidx] = TensorRequantPlan(
            storage_index=sidx,
            name=names.get(sidx, f"tensor_{sidx}"),
            numel=int(numels[sidx]),
            levels=int(levels),
        )
    return plans


# ---------------------------------------------------------------------------
# Contest score (canonical, recomputed-from-components per CLAUDE.md)
# ---------------------------------------------------------------------------

CONTEST_UNCOMPRESSED_BYTES = 37_545_489  # |upstream/videos/0.mkv|


def contest_score_from_components(
    *, d_seg: float, d_pose: float, archive_zip_size: int,
    uncompressed_bytes: int = CONTEST_UNCOMPRESSED_BYTES,
) -> dict[str, float]:
    """score = 100*d_seg + sqrt(10*d_pose) + 25*rate ; rate = bytes/uncompressed.

    Recomputed from components per CLAUDE.md "the rounded final_score field lies".
    """
    import math

    rate = float(archive_zip_size) / float(uncompressed_bytes)
    seg_term = 100.0 * float(d_seg)
    pose_term = math.sqrt(10.0 * float(d_pose))
    rate_term = 25.0 * rate
    return {
        "rate": rate,
        "seg_term": seg_term,
        "pose_term": pose_term,
        "rate_term": rate_term,
        "score": seg_term + pose_term + rate_term,
    }


def score_delta_components(
    *, base: Mapping[str, float], cand: Mapping[str, float]
) -> dict[str, float]:
    """Decompose ΔS into seg / pose / rate contributions (cand − base)."""
    return {
        "d_score": float(cand["score"]) - float(base["score"]),
        "d_seg_term": float(cand["seg_term"]) - float(base["seg_term"]),
        "d_pose_term": float(cand["pose_term"]) - float(base["pose_term"]),
        "d_rate_term": float(cand["rate_term"]) - float(base["rate_term"]),
    }
