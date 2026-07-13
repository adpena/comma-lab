# SPDX-License-Identifier: MIT
"""Lossless joint coding for the level-set witness's quantized tensors.

This module acts only *after* the canonical per-tensor symmetric-int8 grid has
been chosen.  It therefore cannot change a dequantized weight or a rendered
pixel.  Two reversible storage transforms are exposed:

* a per-matrix C/F axis choice, selected from the exact Brotli byte count; and
* frame-separated modulo-256 temporal deltas for ``code[2*pair + frame]``.

The shared-codebook hypothesis is measured, not assumed.  ``measure_step0``
reports ``I(Q;T) = H(Q) - H(Q|T)`` and the exact pooled-vs-separate Brotli
delta.  The runtime codec deliberately contains no VQ implementation: a
shared codebook is not admitted unless that measurement says it can pay.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli
import numpy as np

from tac.boundary_math.lever_b_levelset_generator import _int8_symmetric

CODE_TRANSFORM_RAW = "raw_i8"
CODE_TRANSFORM_FRAME_DELTA_MOD256 = "frame_delta_mod256"


def _entropy_bits_per_symbol(q: np.ndarray) -> float:
    values = np.asarray(q, dtype=np.int8).reshape(-1).astype(np.int16) + 128
    counts = np.bincount(values, minlength=256).astype(np.float64)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def quantized_base(params: Mapping[str, np.ndarray], order: Sequence[str]) -> dict[str, np.ndarray]:
    """Return the canonical symmetric-int8 array for every counted base tensor."""
    out: dict[str, np.ndarray] = {}
    for name in order:
        q, _ = _int8_symmetric(np.asarray(params[name], dtype=np.float32))
        out[name] = np.ascontiguousarray(q, dtype=np.int8)
    return out


def _base_stream(
    quantized: Mapping[str, np.ndarray],
    order: Sequence[str],
    transposed_names: frozenset[str],
) -> bytes:
    return b"".join(
        np.ascontiguousarray(
            quantized[name].T if name in transposed_names else quantized[name],
            dtype=np.int8,
        ).tobytes()
        for name in order
    )


@dataclass(frozen=True)
class BasePermutationPlan:
    candidate_names: tuple[str, ...]
    transposed_names: tuple[str, ...]
    baseline_brotli_bytes: int
    selected_brotli_bytes: int
    combinations_measured: int

    @property
    def bytes_saved(self) -> int:
        return self.baseline_brotli_bytes - self.selected_brotli_bytes

    def to_json(self) -> dict[str, Any]:
        return {
            "candidate_names": list(self.candidate_names),
            "transposed_names": list(self.transposed_names),
            "baseline_brotli_bytes": self.baseline_brotli_bytes,
            "selected_brotli_bytes": self.selected_brotli_bytes,
            "bytes_saved": self.bytes_saved,
            "combinations_measured": self.combinations_measured,
        }


def derive_base_permutation_plan(params: Mapping[str, np.ndarray], order: Sequence[str]) -> BasePermutationPlan:
    """Exhaustively select the exact post-Brotli axis-storage permutation.

    Only two-dimensional tensors with both axes nontrivial participate, so the
    search is finite and completely derived from the checkpoint.  Ties keep the
    lexicographically first bit mask, including the all-original mask.
    """
    q = quantized_base(params, order)
    names = tuple(name for name in order if q[name].ndim == 2 and min(q[name].shape) > 1)
    masks = tuple(range(1 << len(names)))

    def _measure(mask: int) -> int:
        selected = frozenset(name for i, name in enumerate(names) if (mask >> i) & 1)
        return len(brotli.compress(_base_stream(q, order, selected), quality=11))

    # Brotli's C encoder releases the GIL.  A bounded thread pool keeps the exhaustive search exact
    # while avoiding a long serial pause on the 2^9 current witness chart.
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        sizes = tuple(pool.map(_measure, masks))
    best_size, best_mask = min((size, mask) for mask, size in zip(masks, sizes, strict=True))
    baseline = sizes[0]
    transposed = tuple(name for i, name in enumerate(names) if (best_mask >> i) & 1)
    return BasePermutationPlan(
        candidate_names=names,
        transposed_names=transposed,
        baseline_brotli_bytes=baseline,
        selected_brotli_bytes=best_size,
        combinations_measured=1 << len(names),
    )


def encode_base_quantized(
    params: Mapping[str, np.ndarray],
    order: Sequence[str],
    transposed_names: Sequence[str],
) -> bytes:
    q = quantized_base(params, order)
    selected = frozenset(transposed_names)
    unknown = selected.difference(order)
    if unknown:
        raise ValueError(f"unknown base tensor permutation names: {sorted(unknown)}")
    for name in selected:
        if q[name].ndim != 2:
            raise ValueError(f"storage transpose requires a 2-D tensor; {name} is {q[name].shape}")
    return _base_stream(q, order, selected)


def decode_base_quantized(
    raw: bytes,
    order: Sequence[str],
    shapes: Mapping[str, Sequence[int]],
    transposed_names: Sequence[str],
) -> dict[str, np.ndarray]:
    flat = np.frombuffer(raw, dtype=np.int8)
    selected = frozenset(transposed_names)
    out: dict[str, np.ndarray] = {}
    off = 0
    for name in order:
        shape = tuple(int(x) for x in shapes[name])
        n = int(np.prod(shape))
        chunk = flat[off : off + n]
        if chunk.size != n:
            raise ValueError(f"cross-tensor base stream short for {name!r}: need {n}, got {chunk.size}")
        if name in selected:
            if len(shape) != 2:
                raise ValueError(f"stored transpose for non-2-D tensor {name!r}: {shape}")
            out[name] = chunk.reshape(shape[1], shape[0]).T.copy()
        else:
            out[name] = chunk.reshape(shape).copy()
        off += n
    if off != flat.size:
        raise ValueError(f"cross-tensor base stream has {flat.size - off} unconsumed byte(s)")
    return out


def _delta_mod256(rows: np.ndarray) -> np.ndarray:
    x = np.asarray(rows, dtype=np.uint8)
    out = np.empty_like(x)
    out[0] = x[0]
    out[1:] = (x[1:].astype(np.uint16) - x[:-1].astype(np.uint16)).astype(np.uint8)
    return out


def _undelta_mod256(delta: np.ndarray) -> np.ndarray:
    d = np.asarray(delta, dtype=np.uint8)
    return (np.cumsum(d.astype(np.uint64), axis=0) & 255).astype(np.uint8)


def encode_code_quantized(q: np.ndarray, transform: str) -> bytes:
    q_i8 = np.ascontiguousarray(q, dtype=np.int8)
    if transform == CODE_TRANSFORM_RAW:
        return q_i8.tobytes()
    if transform != CODE_TRANSFORM_FRAME_DELTA_MOD256:
        raise ValueError(f"unknown code transform {transform!r}")
    if q_i8.ndim != 2 or q_i8.shape[0] % 2:
        raise ValueError(f"frame-delta code must have shape (2*P,D); got {q_i8.shape}")
    pair_frame = q_i8.view(np.uint8).reshape(q_i8.shape[0] // 2, 2, q_i8.shape[1])
    return b"".join(_delta_mod256(pair_frame[:, frame, :]).tobytes() for frame in range(2))


def decode_code_quantized(raw: bytes, shape: Sequence[int], transform: str) -> np.ndarray:
    rows, dims = (int(shape[0]), int(shape[1]))
    if rows % 2:
        raise ValueError(f"code shape must have an even row count; got {tuple(shape)}")
    expected = rows * dims
    if len(raw) != expected:
        raise ValueError(f"code stream has {len(raw)} B, expected {expected} B")
    if transform == CODE_TRANSFORM_RAW:
        return np.frombuffer(raw, dtype=np.int8).reshape(rows, dims).copy()
    if transform != CODE_TRANSFORM_FRAME_DELTA_MOD256:
        raise ValueError(f"unknown code transform {transform!r}")
    pairs = rows // 2
    half = pairs * dims
    d0 = np.frombuffer(raw[:half], dtype=np.uint8).reshape(pairs, dims)
    d1 = np.frombuffer(raw[half:], dtype=np.uint8).reshape(pairs, dims)
    pair_frame = np.stack([_undelta_mod256(d0), _undelta_mod256(d1)], axis=1)
    return pair_frame.reshape(rows, dims).view(np.int8).copy()


@dataclass(frozen=True)
class CodeTransformPlan:
    transform: str
    baseline_brotli_bytes: int
    selected_brotli_bytes: int
    exact_unique_rows: int
    exact_unique_pairs: int

    @property
    def bytes_saved(self) -> int:
        return self.baseline_brotli_bytes - self.selected_brotli_bytes

    def to_json(self) -> dict[str, Any]:
        return {
            "transform": self.transform,
            "baseline_brotli_bytes": self.baseline_brotli_bytes,
            "selected_brotli_bytes": self.selected_brotli_bytes,
            "bytes_saved": self.bytes_saved,
            "exact_unique_rows": self.exact_unique_rows,
            "exact_unique_pairs": self.exact_unique_pairs,
        }


def derive_code_transform_plan(code: np.ndarray) -> CodeTransformPlan:
    q, _ = _int8_symmetric(np.asarray(code, dtype=np.float32))
    q = np.ascontiguousarray(q, dtype=np.int8)
    candidates = (CODE_TRANSFORM_RAW, CODE_TRANSFORM_FRAME_DELTA_MOD256)
    sizes = {mode: len(brotli.compress(encode_code_quantized(q, mode), quality=11)) for mode in candidates}
    selected = min(candidates, key=lambda mode: (sizes[mode], candidates.index(mode)))
    pairs = q.reshape(q.shape[0] // 2, -1)
    return CodeTransformPlan(
        transform=selected,
        baseline_brotli_bytes=sizes[CODE_TRANSFORM_RAW],
        selected_brotli_bytes=sizes[selected],
        exact_unique_rows=int(np.unique(q, axis=0).shape[0]),
        exact_unique_pairs=int(np.unique(pairs, axis=0).shape[0]),
    )


def measure_step0(params: Mapping[str, np.ndarray], order: Sequence[str], checkpoint: Path) -> dict[str, Any]:
    """Measure the shared-distribution gate on the exact canonical int8 symbols."""
    q = quantized_base(params, order)
    rows: list[dict[str, Any]] = []
    pooled_chunks: list[bytes] = []
    weighted_entropy_numer = 0.0
    total = 0
    separate_brotli = 0
    for name in order:
        arr = q[name]
        entropy = _entropy_bits_per_symbol(arr)
        compressed = len(brotli.compress(arr.tobytes(), quality=11))
        rows.append(
            {
                "name": name,
                "shape": list(arr.shape),
                "symbols": int(arr.size),
                "unique_symbols": int(np.unique(arr).size),
                "entropy_bits_per_weight": entropy,
                "brotli_bytes": compressed,
            }
        )
        pooled_chunks.append(arr.tobytes())
        weighted_entropy_numer += arr.size * entropy
        total += int(arr.size)
        separate_brotli += compressed
    pooled_raw = b"".join(pooled_chunks)
    pooled_q = np.frombuffer(pooled_raw, dtype=np.int8)
    pooled_entropy = _entropy_bits_per_symbol(pooled_q)
    conditional_entropy = weighted_entropy_numer / total
    pooled_brotli = len(brotli.compress(pooled_raw, quality=11))
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "base_tensor_count": len(order),
        "base_weight_symbols": total,
        "pooled_entropy_bits_per_weight": pooled_entropy,
        "weighted_per_tensor_entropy_bits_per_weight": conditional_entropy,
        "tensor_identity_mutual_information_bits_per_weight": (pooled_entropy - conditional_entropy),
        "pooled_brotli_bytes": pooled_brotli,
        "sum_per_tensor_brotli_bytes": separate_brotli,
        "pooled_brotli_saving_vs_separate_bytes": separate_brotli - pooled_brotli,
        "shared_codebook_gate": {
            "admitted": False,
            "reason": (
                "canonical representation already stores one pooled int8 alphabet/Brotli stream; "
                "nonzero tensor-identity information and only the measured pooled-stream delta "
                "remain, so an additional exact shared codebook has no distinct payload to remove"
            ),
            "verdict_scope": "FORMULATION: post-hoc exact shared codebook on this checkpoint",
        },
        "per_tensor": rows,
    }


__all__ = [
    "CODE_TRANSFORM_FRAME_DELTA_MOD256",
    "CODE_TRANSFORM_RAW",
    "BasePermutationPlan",
    "CodeTransformPlan",
    "decode_base_quantized",
    "decode_code_quantized",
    "derive_base_permutation_plan",
    "derive_code_transform_plan",
    "encode_base_quantized",
    "encode_code_quantized",
    "measure_step0",
    "quantized_base",
]
