#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Cross-process MLX probe for integer-lowered resize-adjoint accumulation.

This is the decisive local probe for the 2026-07-13 Pythagorean exact-
arithmetic investigation.  It exercises a real operation from render-R: the
duplicate-index accumulation in the transpose of one 384 -> 874 bicubic resize
axis.  The current float32 formulation and a Q15/int32 lowering use identical
indices and cotangents.

Authority: [macOS-MLX research-signal] for on-device cross-process hash
identity; NumPy-fp32/int32 is the numerical reference.  This tool never emits
a contest score and never launches training, evaluation, or a paid job.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / ".omx/research/pythagorean_exact_arithmetic_bitident_probe_20260713.json"
SCHEMA = "pythagorean_exact_arithmetic_bitident_probe.v1"

# Actual vertical axis of the contest-faithful witness render-R.
IN_SIZE = 384
OUT_SIZE = 874
TAPS = 4
LANES = 128 * 3  # one 128-wide RGB slice; independent resize-adjoint lanes
Q_BITS = 15
Q_SCALE = 1 << Q_BITS
SEED = 157
N_PROCESSES = 10
VARIANTS = ("float_atomic", "fixed_q15_int32_atomic")

# Explicit full-chain extension.  The settled v1 one-axis CLI remains the
# default so its existing receipt is never overwritten by an implicit rerun.
FULL_R_SCHEMA = "pythagorean_exact_arithmetic_full_r_n600.v2"
FULL_R_SCOPE = "full-r-n600"
DEFAULT_GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_FULL_R_OUTPUT = (
    REPO / "experiments/results/throughput_authority_ladder_20260714/full_r_adjoint_n600.json"
)
FULL_R_MEMBERS = ("gt_f0.npy", "gt_f1.npy")
FULL_R_PAIR_COUNT = 600
STATE_BITS_BY_BOUNDARY = (7, 7, 7, 5, 5)
INT32_SAFE_LIMIT = int(np.iinfo(np.int32).max)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = f"{contiguous.dtype.str}:{contiguous.shape}".encode("ascii")
    return _sha256_bytes(header + contiguous.tobytes())


def bicubic_indices_weights_numpy(*, in_size: int = IN_SIZE, out_size: int = OUT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the MLX reference-R align_corners=False cubic coordinate map."""

    scale = np.float32(float(in_size) / float(out_size))
    out = np.arange(out_size, dtype=np.float32)
    real = ((out + np.float32(0.5)) * scale - np.float32(0.5)).astype(np.float32)
    base = np.floor(real).astype(np.int32)
    offsets = np.asarray([-1, 0, 1, 2], dtype=np.int32)
    unclipped = base[:, None] + offsets[None, :]
    distance = real[:, None] - unclipped.astype(np.float32)

    a = np.float32(-0.75)
    x = np.abs(distance).astype(np.float32)
    x2 = (x * x).astype(np.float32)
    x3 = (x2 * x).astype(np.float32)
    inner = ((a + np.float32(2.0)) * x3 - (a + np.float32(3.0)) * x2 + np.float32(1.0)).astype(np.float32)
    outer = (a * x3 - np.float32(5.0) * a * x2 + np.float32(8.0) * a * x - np.float32(4.0) * a).astype(np.float32)
    weights = np.where(
        x <= np.float32(1.0),
        inner,
        np.where(x < np.float32(2.0), outer, np.float32(0.0)),
    ).astype(np.float32)
    indices = np.clip(unclipped, 0, in_size - 1).astype(np.int32)
    return indices, weights


def build_resize_adjoint_fixture(*, seed: int = SEED, lanes: int = LANES) -> dict[str, np.ndarray | float | int]:
    """Build common float/fixed-point operands and deterministic NumPy references."""

    indices, weights = bicubic_indices_weights_numpy()
    rng = np.random.default_rng(seed)
    cotangent_i16 = rng.integers(-127, 128, size=(OUT_SIZE, lanes), dtype=np.int16)
    cotangent_f32 = cotangent_i16.astype(np.float32)

    lane_ids = np.arange(lanes, dtype=np.int32)
    destination = (
        indices[:, :, None].astype(np.int64) * np.int64(lanes) + lane_ids[None, None, :].astype(np.int64)
    ).reshape(-1)
    destination_u32 = destination.astype(np.uint32)

    float_contrib = (weights[:, :, None] * cotangent_f32[:, None, :]).astype(np.float32)
    float_reference = np.zeros(IN_SIZE * lanes, dtype=np.float32)
    np.add.at(float_reference, destination, float_contrib.reshape(-1))

    weights_q15 = np.rint(weights.astype(np.float64) * float(Q_SCALE)).astype(np.int32)
    int_contrib_i64 = weights_q15[:, :, None].astype(np.int64) * cotangent_i16[:, None, :].astype(np.int64)
    int_reference_i64 = np.zeros(IN_SIZE * lanes, dtype=np.int64)
    np.add.at(int_reference_i64, destination, int_contrib_i64.reshape(-1))

    abs_accum_i64 = np.zeros(IN_SIZE * lanes, dtype=np.int64)
    np.add.at(abs_accum_i64, destination, np.abs(int_contrib_i64).reshape(-1))
    max_abs_integer_accumulator = int(np.max(abs_accum_i64))
    if max_abs_integer_accumulator > np.iinfo(np.int32).max:
        raise OverflowError(
            f"Q15 resize-adjoint fixture exceeds int32 accumulator range: {max_abs_integer_accumulator}"
        )
    int_reference = int_reference_i64.astype(np.int32)

    # A priori parity envelope for Q15 lowering versus a float32 fixed-order
    # authority.  Quantization error is summed exactly per destination.  The
    # floating reduction term uses the standard gamma_n bound twice (one order
    # for NumPy, one arbitrary atomic order for MLX).
    dequantized_weights = weights_q15.astype(np.float64) / float(Q_SCALE)
    per_term_quant_error = (
        np.abs(dequantized_weights - weights.astype(np.float64))[:, :, None]
        * np.abs(cotangent_i16.astype(np.float64))[:, None, :]
    )
    quant_bound = np.zeros(IN_SIZE * lanes, dtype=np.float64)
    np.add.at(quant_bound, destination, per_term_quant_error.reshape(-1))

    abs_float_terms = np.zeros(IN_SIZE * lanes, dtype=np.float64)
    np.add.at(abs_float_terms, destination, np.abs(float_contrib.astype(np.float64)).reshape(-1))
    counts = np.zeros(IN_SIZE * lanes, dtype=np.int32)
    np.add.at(counts, destination, 1)
    eps = float(np.finfo(np.float32).eps)
    max_count = int(np.max(counts))
    gamma = (max_count * eps) / (1.0 - max_count * eps)
    rounding_bound = 2.0 * gamma * abs_float_terms
    float_reorder_tolerance = float(np.max(rounding_bound) + 8.0 * eps)
    authority_tolerance = float(np.max(quant_bound + rounding_bound) + 8.0 * eps)

    return {
        "destination_u32": destination_u32,
        "float_contrib": float_contrib.reshape(-1),
        "int_contrib": int_contrib_i64.astype(np.int32).reshape(-1),
        "float_reference": float_reference,
        "int_reference": int_reference,
        "authority_tolerance": authority_tolerance,
        "float_reorder_tolerance": float_reorder_tolerance,
        "max_abs_integer_accumulator": max_abs_integer_accumulator,
        "max_contributions_per_destination": max_count,
    }


def resize_indices_weights_full(
    *, in_size: int, out_size: int, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Contest-R align-corners-false taps for bilinear or a=-0.75 bicubic."""

    normalized = str(mode).strip().lower()
    if normalized == "bicubic":
        return bicubic_indices_weights_numpy(in_size=in_size, out_size=out_size)
    if normalized != "bilinear":
        raise ValueError(f"unsupported resize mode {mode!r}")
    scale = np.float32(float(in_size) / float(out_size))
    out = np.arange(int(out_size), dtype=np.float32)
    real = ((out + np.float32(0.5)) * scale - np.float32(0.5)).astype(np.float32)
    base = np.floor(real).astype(np.int32)
    right = (real - base.astype(np.float32)).astype(np.float32)
    indices = np.stack([base, base + 1], axis=1)
    weights = np.stack([np.float32(1.0) - right, right], axis=1).astype(np.float32)
    return np.clip(indices, 0, int(in_size) - 1).astype(np.int32), weights


def _transpose_plan(*, name: str, in_size: int, out_size: int, mode: str) -> dict[str, Any]:
    """Build original-tap CSR rows for the transpose; duplicates stay distinct."""

    indices, weights = resize_indices_weights_full(
        in_size=int(in_size), out_size=int(out_size), mode=mode
    )
    rows: list[list[tuple[int, int, np.float32]]] = [[] for _ in range(int(in_size))]
    for out_index in range(int(out_size)):
        for tap_index in range(int(indices.shape[1])):
            rows[int(indices[out_index, tap_index])].append(
                (out_index, tap_index, np.float32(weights[out_index, tap_index]))
            )
    max_fan_in = max((len(row) for row in rows), default=0)
    source_pad = np.zeros((int(in_size), max_fan_in), dtype=np.int32)
    float_weight_pad = np.zeros((int(in_size), max_fan_in), dtype=np.float32)
    q_weight_pad = np.zeros((int(in_size), max_fan_in), dtype=np.int32)
    counts = np.zeros((int(in_size),), dtype=np.int32)
    for destination_index, row in enumerate(rows):
        counts[destination_index] = len(row)
        for slot, (source_index, _tap_index, weight) in enumerate(row):
            source_pad[destination_index, slot] = source_index
            float_weight_pad[destination_index, slot] = weight
            q_weight_pad[destination_index, slot] = int(
                np.rint(np.float64(weight) * np.float64(Q_SCALE))
            )
    float_abs_sum = np.sum(np.abs(float_weight_pad.astype(np.float64)), axis=1)
    q_abs_sum = np.sum(np.abs(q_weight_pad.astype(np.int64)), axis=1)
    quant_l1_error = np.sum(
        np.abs(q_weight_pad.astype(np.float64) / float(Q_SCALE) - float_weight_pad),
        axis=1,
    )
    return {
        "name": name,
        "in_size": int(in_size),
        "out_size": int(out_size),
        "mode": str(mode),
        "indices": indices,
        "weights": weights,
        "weights_q15": np.rint(weights.astype(np.float64) * float(Q_SCALE)).astype(np.int32),
        "source_pad": source_pad,
        "float_weight_pad": float_weight_pad,
        "q_weight_pad": q_weight_pad,
        "counts": counts,
        "max_fan_in": int(max_fan_in),
        "max_float_abs_weight_sum": float(np.max(float_abs_sum)),
        "max_q15_abs_weight_sum": int(np.max(q_abs_sum)),
        "max_q15_l1_error": float(np.max(quant_l1_error)),
    }


def build_full_r_plans() -> tuple[dict[str, Any], ...]:
    """The exact reverse-order four-axis R-adjoint chain."""

    return (
        _transpose_plan(
            name="down_w_transpose_512_to_1164",
            in_size=1164,
            out_size=512,
            mode="bilinear",
        ),
        _transpose_plan(
            name="down_h_transpose_384_to_874",
            in_size=874,
            out_size=384,
            mode="bilinear",
        ),
        _transpose_plan(
            name="up_w_transpose_1164_to_512",
            in_size=512,
            out_size=1164,
            mode="bicubic",
        ),
        _transpose_plan(
            name="up_h_transpose_874_to_384",
            in_size=384,
            out_size=874,
            mode="bicubic",
        ),
    )


def signed_round_divide(values: np.ndarray, divisor: int) -> np.ndarray:
    """Round signed integers to nearest, ties away from zero, without signed shifts."""

    divisor = int(divisor)
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    source = np.asarray(values, dtype=np.int64)
    magnitude = np.abs(source)
    rounded = (magnitude + divisor // 2) // divisor
    result = np.where(source < 0, -rounded, rounded)
    if result.size and (
        int(np.min(result)) < np.iinfo(np.int32).min
        or int(np.max(result)) > np.iinfo(np.int32).max
    ):
        raise OverflowError("signed requantization result exceeds int32")
    return result.astype(np.int32)


def minimum_signed_bits_for_bound(bound: int) -> int:
    """Minimum two's-complement width covering every integer in [-bound,+bound]."""

    bound = int(bound)
    if bound < 0:
        raise ValueError("bound must be non-negative")
    return max(1, int(np.ceil(np.log2(2 * bound + 1)))) if bound else 1


def integer_stage_preflight(
    plan: dict[str, Any], *, max_abs_input: int, in_state_bits: int, out_state_bits: int
) -> dict[str, Any]:
    """A priori int32 sum-of-absolute-contributions proof for one axis."""

    max_abs_input = int(max_abs_input)
    if max_abs_input < 0:
        raise ValueError("max_abs_input must be non-negative")
    q_abs_sum = int(plan["max_q15_abs_weight_sum"])
    accumulator_bound = max_abs_input * q_abs_sum
    shift = Q_BITS + int(in_state_bits) - int(out_state_bits)
    if shift < 0:
        raise ValueError("requantization cannot left-shift the accumulator")
    divisor = 1 << shift
    safe_limit = INT32_SAFE_LIMIT - divisor // 2
    safe = accumulator_bound <= safe_limit
    output_bound = (accumulator_bound + divisor // 2) // divisor
    result = {
        "stage": str(plan["name"]),
        "bound_kind": "STATIC_WORST_CASE_LINF",
        "max_fan_in": int(plan["max_fan_in"]),
        "max_abs_input_integer": max_abs_input,
        "max_q15_abs_weight_sum": q_abs_sum,
        "max_abs_accumulator_bound": int(accumulator_bound),
        "minimum_signed_accumulator_bits": minimum_signed_bits_for_bound(
            accumulator_bound
        ),
        "implemented_accumulator_bits": 32,
        "int32_safe_limit_after_rounding_headroom": int(safe_limit),
        "int32_headroom_x": (
            float(safe_limit) / float(accumulator_bound) if accumulator_bound else float("inf")
        ),
        "in_state_bits": int(in_state_bits),
        "out_state_bits": int(out_state_bits),
        "requant_divisor": int(divisor),
        "max_abs_output_integer_bound": int(output_bound),
        "safe": bool(safe),
        "reorder_identity_preconditions": (
            "exact integer add over every reachable partial sum; no overflow or saturation; "
            "one deterministic signed requantization"
        ),
    }
    if not safe:
        raise OverflowError(
            f"{plan['name']} int32 proof failed: {accumulator_bound} > {safe_limit}"
        )
    return result


def full_r_static_overflow_proof(
    plans: tuple[dict[str, Any], ...] | None = None,
) -> list[dict[str, Any]]:
    """Prove the preregistered Q15 + boundary-Q schedule for |cotangent|<=127."""

    selected = plans if plans is not None else build_full_r_plans()
    maximum = 127 * (1 << STATE_BITS_BY_BOUNDARY[0])
    proofs: list[dict[str, Any]] = []
    for index, plan in enumerate(selected):
        proof = integer_stage_preflight(
            plan,
            max_abs_input=maximum,
            in_state_bits=STATE_BITS_BY_BOUNDARY[index],
            out_state_bits=STATE_BITS_BY_BOUNDARY[index + 1],
        )
        proofs.append(proof)
        maximum = int(proof["max_abs_output_integer_bound"])
    return proofs


def _transpose_axis_numpy_float(inp_l_sout_r: np.ndarray, plan: dict[str, Any]) -> np.ndarray:
    inp = np.asarray(inp_l_sout_r, dtype=np.float32)
    if inp.ndim != 3 or int(inp.shape[1]) != int(plan["out_size"]):
        raise ValueError(
            f"{plan['name']} expects (L,{plan['out_size']},R), got {inp.shape}"
        )
    output = np.zeros((inp.shape[0], int(plan["in_size"]), inp.shape[2]), dtype=np.float32)
    source = np.asarray(plan["source_pad"], dtype=np.int32)
    weights = np.asarray(plan["float_weight_pad"], dtype=np.float32)
    for slot in range(int(plan["max_fan_in"])):
        term = (
            inp[:, source[:, slot], :]
            * weights[:, slot].reshape(1, -1, 1)
        ).astype(np.float32)
        output = (output + term).astype(np.float32)
    return output


def _transpose_axis_numpy_integer(
    inp_l_sout_r: np.ndarray,
    plan: dict[str, Any],
    *,
    in_state_bits: int,
    out_state_bits: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    inp = np.asarray(inp_l_sout_r, dtype=np.int32)
    if inp.ndim != 3 or int(inp.shape[1]) != int(plan["out_size"]):
        raise ValueError(
            f"{plan['name']} expects (L,{plan['out_size']},R), got {inp.shape}"
        )
    proof = integer_stage_preflight(
        plan,
        max_abs_input=int(np.max(np.abs(inp.astype(np.int64)))) if inp.size else 0,
        in_state_bits=in_state_bits,
        out_state_bits=out_state_bits,
    )
    accumulator = np.zeros(
        (inp.shape[0], int(plan["in_size"]), inp.shape[2]), dtype=np.int64
    )
    absolute = np.zeros_like(accumulator)
    source = np.asarray(plan["source_pad"], dtype=np.int32)
    weights = np.asarray(plan["q_weight_pad"], dtype=np.int64)
    for slot in range(int(plan["max_fan_in"])):
        term = inp[:, source[:, slot], :].astype(np.int64) * weights[
            :, slot
        ].reshape(1, -1, 1)
        accumulator += term
        absolute += np.abs(term)
    actual_abs_bound = int(np.max(absolute)) if absolute.size else 0
    if actual_abs_bound > INT32_SAFE_LIMIT:
        raise OverflowError(
            f"{plan['name']} actual sum-of-absolute-contributions exceeds int32: "
            f"{actual_abs_bound}"
        )
    shift = Q_BITS + int(in_state_bits) - int(out_state_bits)
    proof["actual_max_sum_abs_contributions"] = actual_abs_bound
    proof["actual_bound_kind"] = "OBSERVED_FRAME_SUM_ABS_CONTRIBUTIONS"
    proof["actual_minimum_signed_accumulator_bits"] = minimum_signed_bits_for_bound(
        actual_abs_bound
    )
    proof["actual_int32_headroom_x"] = (
        float(INT32_SAFE_LIMIT) / float(actual_abs_bound) if actual_abs_bound else float("inf")
    )
    return signed_round_divide(accumulator, 1 << shift), proof


def _resample_2d_numpy(
    x_hwc: np.ndarray, *, out_h: int, out_w: int, mode: str
) -> np.ndarray:
    x = np.asarray(x_hwc, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB, got {x.shape}")
    hidx, hw = resize_indices_weights_full(
        in_size=int(x.shape[0]), out_size=int(out_h), mode=mode
    )
    widx, ww = resize_indices_weights_full(
        in_size=int(x.shape[1]), out_size=int(out_w), mode=mode
    )
    gathered_h = x[hidx, :, :]
    intermediate = np.sum(
        gathered_h * hw[:, :, None, None], axis=1, dtype=np.float32
    ).astype(np.float32)
    gathered_w = intermediate[:, widx, :]
    return np.sum(
        gathered_w * ww[None, :, :, None], axis=2, dtype=np.float32
    ).astype(np.float32)


def prepare_real_full_r_fixture(frame_hwc: np.ndarray) -> dict[str, np.ndarray]:
    """Derive the preregistered real-frame low image, mask, and residual cotangent."""

    frame = np.asarray(frame_hwc)
    if frame.shape != (874, 1164, 3) or frame.dtype != np.uint8:
        raise ValueError(f"expected uint8 (874,1164,3), got {frame.dtype} {frame.shape}")
    low = _resample_2d_numpy(frame, out_h=384, out_w=512, mode="bilinear")
    camera_pre_round = _resample_2d_numpy(low, out_h=874, out_w=1164, mode="bicubic")
    mask = np.logical_and(camera_pre_round > 0.0, camera_pre_round < 255.0)
    cotangent = np.clip(np.rint(low - np.float32(127.5)), -127, 127).astype(np.int32)
    return {"low": low, "clip_mask": mask, "cotangent": cotangent}


def full_r_vjp_numpy_float(
    cotangent_hwc: np.ndarray,
    clip_mask_hwc: np.ndarray,
    *,
    plans: tuple[dict[str, Any], ...] | None = None,
) -> np.ndarray:
    selected = plans if plans is not None else build_full_r_plans()
    value = np.asarray(cotangent_hwc, dtype=np.float32)
    if value.shape != (384, 512, 3):
        raise ValueError(f"expected scorer cotangent (384,512,3), got {value.shape}")
    value = _transpose_axis_numpy_float(value.reshape(384, 512, 3), selected[0])
    value = _transpose_axis_numpy_float(value.reshape(1, 384, 1164 * 3), selected[1])
    value = value.reshape(874, 1164, 3)
    value = (value * np.asarray(clip_mask_hwc, dtype=np.float32)).astype(np.float32)
    value = _transpose_axis_numpy_float(value.reshape(874, 1164, 3), selected[2])
    value = _transpose_axis_numpy_float(value.reshape(1, 874, 512 * 3), selected[3])
    return value.reshape(384, 512, 3)


def full_r_vjp_numpy_integer(
    cotangent_hwc: np.ndarray,
    clip_mask_hwc: np.ndarray,
    *,
    plans: tuple[dict[str, Any], ...] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    selected = plans if plans is not None else build_full_r_plans()
    base = np.asarray(cotangent_hwc, dtype=np.int32)
    if base.shape != (384, 512, 3):
        raise ValueError(f"expected scorer cotangent (384,512,3), got {base.shape}")
    value = base * np.int32(1 << STATE_BITS_BY_BOUNDARY[0])
    proofs: list[dict[str, Any]] = []
    value, proof = _transpose_axis_numpy_integer(
        value.reshape(384, 512, 3),
        selected[0],
        in_state_bits=STATE_BITS_BY_BOUNDARY[0],
        out_state_bits=STATE_BITS_BY_BOUNDARY[1],
    )
    proofs.append(proof)
    value, proof = _transpose_axis_numpy_integer(
        value.reshape(1, 384, 1164 * 3),
        selected[1],
        in_state_bits=STATE_BITS_BY_BOUNDARY[1],
        out_state_bits=STATE_BITS_BY_BOUNDARY[2],
    )
    proofs.append(proof)
    value = value.reshape(874, 1164, 3)
    value = value * np.asarray(clip_mask_hwc, dtype=np.int32)
    value, proof = _transpose_axis_numpy_integer(
        value.reshape(874, 1164, 3),
        selected[2],
        in_state_bits=STATE_BITS_BY_BOUNDARY[2],
        out_state_bits=STATE_BITS_BY_BOUNDARY[3],
    )
    proofs.append(proof)
    value, proof = _transpose_axis_numpy_integer(
        value.reshape(1, 874, 512 * 3),
        selected[3],
        in_state_bits=STATE_BITS_BY_BOUNDARY[3],
        out_state_bits=STATE_BITS_BY_BOUNDARY[4],
    )
    proofs.append(proof)
    return value.reshape(384, 512, 3), proofs


_MLX_SCATTER_CACHE: dict[tuple[str, int, int, str], tuple[Any, Any, Any]] = {}
_MLX_REQUANT_KERNEL: Any | None = None


def _scatter_vectors_numpy(
    plan: dict[str, Any], *, left: int, right: int, integer: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten original forward taps into source/destination/value vectors."""

    left = int(left)
    right = int(right)
    sout = int(plan["out_size"])
    sin = int(plan["in_size"])
    taps = int(np.asarray(plan["indices"]).shape[1])
    source = np.broadcast_to(
        np.arange(left * sout * right, dtype=np.uint32).reshape(left, sout, 1, right),
        (left, sout, taps, right),
    ).reshape(-1)
    destination = (
        np.arange(left, dtype=np.uint64)[:, None, None, None] * np.uint64(sin * right)
        + np.asarray(plan["indices"], dtype=np.uint64)[None, :, :, None]
        * np.uint64(right)
        + np.arange(right, dtype=np.uint64)[None, None, None, :]
    ).reshape(-1)
    if destination.size and int(np.max(destination)) > np.iinfo(np.uint32).max:
        raise OverflowError("scatter destination exceeds uint32 indexing")
    weights_source = plan["weights_q15"] if integer else plan["weights"]
    weights = np.broadcast_to(
        np.asarray(weights_source)[None, :, :, None],
        (left, sout, taps, right),
    ).reshape(-1)
    return (
        np.ascontiguousarray(source, dtype=np.uint32),
        np.ascontiguousarray(destination, dtype=np.uint32),
        np.ascontiguousarray(weights, dtype=np.int32 if integer else np.float32),
    )


def _mlx_scatter_packet(
    plan: dict[str, Any], *, left: int, right: int, integer: bool
) -> tuple[Any, Any, Any]:
    import mlx.core as mx

    key = (str(plan["name"]), int(left), int(right), "int" if integer else "float")
    cached = _MLX_SCATTER_CACHE.get(key)
    if cached is None:
        source, destination, weights = _scatter_vectors_numpy(
            plan, left=left, right=right, integer=integer
        )
        cached = (
            mx.array(source, dtype=mx.uint32),
            mx.array(destination, dtype=mx.uint32),
            mx.array(weights, dtype=mx.int32 if integer else mx.float32),
        )
        _MLX_SCATTER_CACHE[key] = cached
    return cached


def _mlx_signed_requantize(values: Any, *, divisor: int) -> Any:
    import mlx.core as mx

    global _MLX_REQUANT_KERNEL
    if _MLX_REQUANT_KERNEL is None:
        _MLX_REQUANT_KERNEL = mx.fast.metal_kernel(
            name="full_r_signed_round_nearest_away_i32",
            input_names=["inp", "dims"],
            output_names=["out"],
            source="""
                uint gid = thread_position_in_grid.x;
                int count = dims[0];
                int divisor = dims[1];
                if (gid >= (uint)count) return;
                int value = inp[gid];
                int magnitude = value < 0 ? -value : value;
                int rounded = (magnitude + divisor / 2) / divisor;
                out[gid] = value < 0 ? -rounded : rounded;
            """,
        )
    count = int(values.size)
    dims = mx.array([count, int(divisor)], dtype=mx.int32)
    (output,) = _MLX_REQUANT_KERNEL(
        inputs=[values, dims],
        output_shapes=[values.shape],
        output_dtypes=[mx.int32],
        grid=(count, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return output


def _transpose_axis_mlx_atomic(
    inp_l_sout_r: Any,
    plan: dict[str, Any],
    *,
    integer: bool,
    in_state_bits: int | None = None,
    out_state_bits: int | None = None,
) -> Any:
    import mlx.core as mx

    left, sout, right = map(int, inp_l_sout_r.shape)
    if sout != int(plan["out_size"]):
        raise ValueError(f"{plan['name']} expected Sout={plan['out_size']}, got {sout}")
    source, destination, weights = _mlx_scatter_packet(
        plan, left=left, right=right, integer=integer
    )
    flat = mx.reshape(inp_l_sout_r, (-1,))
    contribution = flat[source] * weights
    output = mx.zeros((left * int(plan["in_size"]) * right,), dtype=contribution.dtype)
    output = output.at[destination].add(contribution)
    if integer:
        if in_state_bits is None or out_state_bits is None:
            raise ValueError("integer atomic transpose requires state-bit boundaries")
        shift = Q_BITS + int(in_state_bits) - int(out_state_bits)
        output = _mlx_signed_requantize(output, divisor=1 << shift)
    return mx.reshape(output, (left, int(plan["in_size"]), right))


def full_r_vjp_mlx_atomic(
    cotangent_hwc: np.ndarray,
    clip_mask_hwc: np.ndarray,
    *,
    variant: str,
    plans: tuple[dict[str, Any], ...] | None = None,
) -> np.ndarray:
    """Run the whole four-axis chain with float or integer MLX atomic scatters."""

    if variant not in VARIANTS:
        raise ValueError(f"unknown full-R variant {variant!r}")
    import mlx.core as mx

    selected = plans if plans is not None else build_full_r_plans()
    integer = variant == "fixed_q15_int32_atomic"
    if integer:
        static_proofs = full_r_static_overflow_proof(selected)
        if not all(bool(row["safe"]) for row in static_proofs):
            raise OverflowError("full-R static integer proof failed")
        value = mx.array(
            np.asarray(cotangent_hwc, dtype=np.int32)
            * np.int32(1 << STATE_BITS_BY_BOUNDARY[0]),
            dtype=mx.int32,
        )
    else:
        value = mx.array(cotangent_hwc, dtype=mx.float32)
    value = _transpose_axis_mlx_atomic(
        mx.reshape(value, (384, 512, 3)),
        selected[0],
        integer=integer,
        in_state_bits=STATE_BITS_BY_BOUNDARY[0],
        out_state_bits=STATE_BITS_BY_BOUNDARY[1],
    )
    value = _transpose_axis_mlx_atomic(
        mx.reshape(value, (1, 384, 1164 * 3)),
        selected[1],
        integer=integer,
        in_state_bits=STATE_BITS_BY_BOUNDARY[1],
        out_state_bits=STATE_BITS_BY_BOUNDARY[2],
    )
    value = mx.reshape(value, (874, 1164, 3))
    value = value * mx.array(
        np.asarray(clip_mask_hwc, dtype=np.int32 if integer else np.float32),
        dtype=mx.int32 if integer else mx.float32,
    )
    value = _transpose_axis_mlx_atomic(
        mx.reshape(value, (874, 1164, 3)),
        selected[2],
        integer=integer,
        in_state_bits=STATE_BITS_BY_BOUNDARY[2],
        out_state_bits=STATE_BITS_BY_BOUNDARY[3],
    )
    value = _transpose_axis_mlx_atomic(
        mx.reshape(value, (1, 874, 512 * 3)),
        selected[3],
        integer=integer,
        in_state_bits=STATE_BITS_BY_BOUNDARY[3],
        out_state_bits=STATE_BITS_BY_BOUNDARY[4],
    )
    mx.eval(value)
    dtype = np.int32 if integer else np.float32
    return np.asarray(value, dtype=dtype).reshape(384, 512, 3)


def _stored_member(gt_cache: Path, member: str) -> np.memmap:
    src = REPO / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from tac.local_acceleration.ane_unlock_followup_20260713 import stored_npy_memmap

    return stored_npy_memmap(gt_cache, member)


def _ordered_real_frames(
    *, gt_cache: Path, pair_start: int, pair_count: int
) -> Any:
    members = {member: _stored_member(gt_cache, member) for member in FULL_R_MEMBERS}
    expected_shape = (FULL_R_PAIR_COUNT, 874, 1164, 3)
    for member, array in members.items():
        if tuple(array.shape) != expected_shape or array.dtype != np.uint8:
            raise ValueError(
                f"{member} custody mismatch: expected uint8 {expected_shape}, "
                f"got {array.dtype} {array.shape}"
            )
    stop = int(pair_start) + int(pair_count)
    if pair_start < 0 or pair_count <= 0 or stop > FULL_R_PAIR_COUNT:
        raise ValueError(f"invalid pair interval [{pair_start},{stop})")
    for pair_index in range(int(pair_start), stop):
        for member in FULL_R_MEMBERS:
            yield pair_index, member, np.asarray(members[member][pair_index])


def _update_corpus_digest(
    digest: Any, *, pair_index: int, member: str, output_sha256: str
) -> None:
    digest.update(f"{pair_index}:{member}:{output_sha256}\n".encode("ascii"))


def derive_full_r_integer_error_bound(
    plans: tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Any]:
    selected = plans if plans is not None else build_full_r_plans()
    magnitude = 127.0
    error = 0.0
    rows: list[dict[str, Any]] = []
    for index, plan in enumerate(selected):
        norm = float(plan["max_float_abs_weight_sum"])
        weight_error = float(plan["max_q15_l1_error"])
        output_bits = int(STATE_BITS_BY_BOUNDARY[index + 1])
        propagated = norm * error
        coefficient = magnitude * weight_error
        requant = 0.5 / float(1 << output_bits)
        error = propagated + coefficient + requant
        magnitude = norm * magnitude
        rows.append(
            {
                "stage": str(plan["name"]),
                "operator_linf_norm": norm,
                "q15_weight_l1_error": weight_error,
                "propagated_input_error": propagated,
                "coefficient_quantization_error_bound": coefficient,
                "requantization_error_bound": requant,
                "output_error_bound": error,
                "output_magnitude_bound": magnitude,
            }
        )
    return {
        "derivation": (
            "e_next <= ||T||_inf*e + ||x||_inf*||Tq-T||_inf + "
            "0.5/2^output_state_bits"
        ),
        "stages": rows,
        "final_max_abs_error_bound": float(error),
    }


def full_r_precision_manifest(
    plans: tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Any]:
    selected = plans if plans is not None else build_full_r_plans()
    proof = full_r_static_overflow_proof(selected)
    error = derive_full_r_integer_error_bound(selected)
    return {
        "schema": "frontier_math_precision_manifest.v1",
        "bound_kind": "STATIC_WORST_CASE_LINF",
        "layers": [
            {
                "op": row["stage"],
                "options": [
                    {
                        "weight_bits": Q_BITS,
                        "input_state_bits": row["in_state_bits"],
                        "output_state_bits": row["out_state_bits"],
                        "accumulator_bits": 32,
                        "minimum_exact_signed_accumulator_bits": row[
                            "minimum_signed_accumulator_bits"
                        ],
                        "error_bound": error["stages"][index]["output_error_bound"],
                        "measured_cost": None,
                        "measured_cost_status": "OWED_HOST_METAL",
                    }
                ],
            }
            for index, row in enumerate(proof)
        ],
        "family_error_budget": error["final_max_abs_error_bound"],
        "observed_corpus_is_not_unseen_input_ibp": True,
    }


def run_full_mlx_variant(
    *, variant: str, gt_cache: Path, pair_start: int, pair_count: int
) -> dict[str, Any]:
    """One cross-process full-corpus Metal cell; return digests, never tensors."""

    started = time.perf_counter()
    try:
        import mlx.core as mx

        mx.set_default_device(mx.gpu)
        probe = mx.sum(mx.arange(8, dtype=mx.float32))
        mx.eval(probe)
        if float(probe.item()) != 28.0:
            raise RuntimeError("evaluated Metal allocation probe returned the wrong value")
    except Exception as exc:
        return {
            "variant": variant,
            "status": "BLOCKED_NOT_MEASURED",
            "blocker": f"evaluated Metal device unavailable: {type(exc).__name__}: {exc}",
            "verdict_scope": "ENVIRONMENT: this process has no evaluated MLX Metal device",
        }
    plans = build_full_r_plans()
    static_proofs = full_r_static_overflow_proof(plans)
    corpus = hashlib.sha256()
    frames = 0
    for pair_index, member, frame in _ordered_real_frames(
        gt_cache=gt_cache, pair_start=pair_start, pair_count=pair_count
    ):
        fixture = prepare_real_full_r_fixture(frame)
        output = full_r_vjp_mlx_atomic(
            fixture["cotangent"], fixture["clip_mask"], variant=variant, plans=plans
        )
        output_sha = _hash_array(output)
        _update_corpus_digest(
            corpus, pair_index=pair_index, member=member, output_sha256=output_sha
        )
        frames += 1
    return {
        "variant": variant,
        "status": "MEASURED",
        "corpus_sha256": corpus.hexdigest(),
        "frames": int(frames),
        "pairs": int(pair_count),
        "pair_start": int(pair_start),
        "members": list(FULL_R_MEMBERS),
        "elapsed_seconds": float(time.perf_counter() - started),
        "device": str(mx.default_device()),
        "mlx_version": importlib.metadata.version("mlx"),
        "static_overflow_proofs": static_proofs if variant == VARIANTS[1] else None,
    }


def run_mlx_variant(variant: str) -> dict[str, Any]:
    """Execute one child cell on MLX GPU and compare it to NumPy authority."""

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    fixture = build_resize_adjoint_fixture()
    destination = mx.array(fixture["destination_u32"], dtype=mx.uint32)
    output_size = IN_SIZE * LANES

    if variant == "float_atomic":
        contribution = mx.array(fixture["float_contrib"], dtype=mx.float32)
        output_mx = mx.zeros((output_size,), dtype=mx.float32).at[destination].add(contribution)
        mx.eval(output_mx)
        output = np.asarray(output_mx, dtype=np.float32)
        reference = np.asarray(fixture["float_reference"], dtype=np.float32)
        max_abs_error = float(np.max(np.abs(output.astype(np.float64) - reference.astype(np.float64))))
        parity = {
            "reference_dtype": "numpy-fp32",
            "bit_identical": bool(np.array_equal(output.view(np.uint32), reference.view(np.uint32))),
            "max_abs_error": max_abs_error,
            "derived_two_order_fp32_tolerance": float(fixture["float_reorder_tolerance"]),
            "within_derived_tolerance": bool(max_abs_error <= float(fixture["float_reorder_tolerance"])),
            "interpretation": "fp32 atomic reduction-order delta; not authority bytes",
        }
    else:
        contribution = mx.array(fixture["int_contrib"], dtype=mx.int32)
        output_mx = mx.zeros((output_size,), dtype=mx.int32).at[destination].add(contribution)
        mx.eval(output_mx)
        output = np.asarray(output_mx, dtype=np.int32)
        reference = np.asarray(fixture["int_reference"], dtype=np.int32)
        dequantized = output.astype(np.float64) / float(Q_SCALE)
        float_reference = np.asarray(fixture["float_reference"], dtype=np.float32).astype(np.float64)
        max_abs_error = float(np.max(np.abs(dequantized - float_reference)))
        tolerance = float(fixture["authority_tolerance"])
        parity = {
            "reference_dtype": "numpy-int32 plus numpy-fp32 resize-adjoint authority",
            "integer_bit_identical": bool(np.array_equal(output, reference)),
            "dequantized_max_abs_error_vs_numpy_fp32": max_abs_error,
            "derived_quantization_plus_fp32_tolerance": tolerance,
            "within_derived_tolerance": bool(max_abs_error <= tolerance),
        }

    return {
        "variant": variant,
        "output_sha256": _hash_array(output),
        "output_dtype": str(output.dtype),
        "output_shape": list(output.shape),
        "parity": parity,
        "mlx_version": importlib.metadata.version("mlx"),
        "device": str(mx.default_device()),
        "fixture": {
            "in_size": IN_SIZE,
            "out_size": OUT_SIZE,
            "taps": TAPS,
            "lanes": LANES,
            "q_bits": Q_BITS,
            "seed": SEED,
            "max_abs_integer_accumulator": int(fixture["max_abs_integer_accumulator"]),
            "int32_limit": int(np.iinfo(np.int32).max),
            "max_contributions_per_destination": int(fixture["max_contributions_per_destination"]),
        },
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _base_receipt(n: int) -> dict[str, Any]:
    fixture = build_resize_adjoint_fixture()
    dequantized = np.asarray(fixture["int_reference"], dtype=np.int32).astype(np.float64) / float(Q_SCALE)
    float_reference = np.asarray(fixture["float_reference"], dtype=np.float32).astype(np.float64)
    error = dequantized - float_reference
    return {
        "schema": SCHEMA,
        "lane_id": "pythagorean_exact_arithmetic_bitident",
        "axis": "[macOS-MLX research-signal; NumPy-fp32/int32 authority; non-promotable MEANS]",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "training": False,
        "paid_dispatch": False,
        "live_run_mutation": False,
        "git_head_at_probe": _git_head(),
        "host": platform.node(),
        "platform": platform.platform(),
        "n_requested_per_variant": n,
        "contract": {
            "real_op": "one-axis transpose accumulation of render-R bicubic resize 384->874",
            "float_formulation": "MLX float32 duplicate-index atomic scatter-add",
            "integer_formulation": "same indices/cotangent; Q15 cubic weights; bounded MLX int32 atomic add",
            "acceptance": (
                "float cross-process divergence; integer 0/N divergence; exact NumPy-int32 parity; "
                "dequantized result within derived Q15+fp32 authority tolerance"
            ),
        },
        "numpy_static_contract": {
            "int_reference_sha256": _hash_array(np.asarray(fixture["int_reference"], dtype=np.int32)),
            "float_reference_sha256": _hash_array(np.asarray(fixture["float_reference"], dtype=np.float32)),
            "max_abs_integer_accumulator": int(fixture["max_abs_integer_accumulator"]),
            "int32_positive_limit": int(np.iinfo(np.int32).max),
            "overflow_headroom_x": float(np.iinfo(np.int32).max) / float(fixture["max_abs_integer_accumulator"]),
            "max_contributions_per_destination": int(fixture["max_contributions_per_destination"]),
            "float_reorder_tolerance": float(fixture["float_reorder_tolerance"]),
            "dequantized_max_abs_error_vs_numpy_fp32": float(np.max(np.abs(error))),
            "dequantized_rmse_vs_numpy_fp32": float(np.sqrt(np.mean(error * error))),
            "derived_quantization_plus_fp32_tolerance": float(fixture["authority_tolerance"]),
            "within_derived_tolerance": bool(float(np.max(np.abs(error))) <= float(fixture["authority_tolerance"])),
        },
        "source_custody": {
            "probe": {
                "path": "tools/probe_pythagorean_exact_arithmetic_bitident.py",
                "sha256": _sha256_file(Path(__file__)),
            },
            "reference_r": {
                "path": "src/tac/local_acceleration/pr95_hnerv_mlx_training.py",
                "sha256": _sha256_file(REPO / "src/tac/local_acceleration/pr95_hnerv_mlx_training.py"),
            },
            "l70_probe": {
                "path": "tools/mlx_gpu_determinism_probe.py",
                "sha256": _sha256_file(REPO / "tools/mlx_gpu_determinism_probe.py"),
            },
        },
        "trials": {variant: [] for variant in VARIANTS},
    }


def _summarize(receipt: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for variant in VARIANTS:
        rows = receipt["trials"][variant]
        hashes = [row["output_sha256"] for row in rows if "output_sha256" in row]
        counts = Counter(hashes)
        representative_count = max(counts.values()) if counts else 0
        entry: dict[str, Any] = {
            "n": len(hashes),
            "unique_hashes": len(counts),
            "cross_process_identical": len(counts) == 1 and len(hashes) > 0,
            "divergent_from_modal_hash": len(hashes) - representative_count,
            "hashes": hashes,
        }
        if rows:
            entry["parity_all"] = all(
                (
                    row["parity"].get("integer_bit_identical", True)
                    and row["parity"].get("within_derived_tolerance", True)
                )
                for row in rows
            )
            entry["max_abs_error_vs_numpy"] = max(
                float(
                    row["parity"].get(
                        "dequantized_max_abs_error_vs_numpy_fp32",
                        row["parity"].get("max_abs_error", 0.0),
                    )
                )
                for row in rows
            )
        summary[variant] = entry

    requested = int(receipt["n_requested_per_variant"])
    complete = all(int(summary[variant]["n"]) == requested for variant in VARIANTS)
    float_diverges = bool(summary["float_atomic"]["n"] > 0 and not summary["float_atomic"]["cross_process_identical"])
    integer_identical = summary["fixed_q15_int32_atomic"]["cross_process_identical"]
    integer_parity = bool(summary["fixed_q15_int32_atomic"].get("parity_all", False))
    decisive_positive = bool(complete and float_diverges and integer_identical and integer_parity)
    if not complete:
        verdict = "INCOMPLETE"
    elif decisive_positive:
        verdict = "REAL-L70-LEVER"
    elif float_diverges and not integer_identical:
        verdict = "L70-DEEPER-THAN-FP-REORDER"
    else:
        verdict = "INERT-CURIO"
    summary["decisive_positive"] = decisive_positive
    summary["complete"] = complete
    summary["overall_verdict"] = verdict
    summary["verdict_scope"] = (
        "INSTANCE x MLX-0.31.2 x M5-Max-Metal x render-R bicubic 384->874 one-axis "
        "transpose accumulation x Q15/int32 bounded fixture"
    )
    return summary


def run_parent(*, output: Path, n: int, resume: bool) -> dict[str, Any]:
    if n < 2:
        raise ValueError("n must be >=2 for a cross-process verdict")
    if resume and output.exists():
        receipt = json.loads(output.read_text(encoding="utf-8"))
        if receipt.get("schema") != SCHEMA:
            raise ValueError(f"cannot resume incompatible receipt {output}")
        if int(receipt.get("n_requested_per_variant", -1)) != n:
            raise ValueError("resume n does not match existing receipt")
        recorded_probe_sha = receipt.get("source_custody", {}).get("probe", {}).get("sha256")
        current_probe_sha = _sha256_file(Path(__file__))
        if recorded_probe_sha != current_probe_sha:
            raise ValueError(
                "resume probe bytes differ from receipt custody; start a fresh receipt "
                f"({recorded_probe_sha!r} != {current_probe_sha!r})"
            )
    else:
        receipt = _base_receipt(n)
        _atomic_write_json(output, receipt)

    receipt.pop("failure", None)
    receipt.pop("completed", None)
    _atomic_write_json(output, receipt)

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    for variant in VARIANTS:
        rows: list[dict[str, Any]] = receipt["trials"][variant]
        while len(rows) < n:
            trial_index = len(rows)
            process = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--child", variant],
                cwd=REPO,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            if process.returncode != 0:
                receipt["failure"] = {
                    "variant": variant,
                    "trial_index": trial_index,
                    "returncode": process.returncode,
                    "stderr_tail": process.stderr[-1200:],
                    "stdout_tail": process.stdout[-1200:],
                }
                _atomic_write_json(output, receipt)
                raise RuntimeError(f"child failed for {variant} trial {trial_index}: {process.stderr[-400:]}")
            row = json.loads(process.stdout.strip().splitlines()[-1])
            row["trial_index"] = trial_index
            rows.append(row)
            receipt["summary"] = _summarize(receipt)
            _atomic_write_json(output, receipt)

    receipt["summary"] = _summarize(receipt)
    receipt["completed"] = True
    _atomic_write_json(output, receipt)
    return receipt


def _full_contract_payload(
    *, gt_cache: Path, cache_sha256: str, pair_start: int, pair_count: int, n: int
) -> dict[str, Any]:
    return {
        "scope": FULL_R_SCOPE,
        "gt_cache": str(gt_cache.resolve()),
        "gt_cache_sha256": cache_sha256,
        "members": list(FULL_R_MEMBERS),
        "pair_start": int(pair_start),
        "pair_count": int(pair_count),
        "frames": int(pair_count) * len(FULL_R_MEMBERS),
        "n_processes_per_variant": int(n),
        "q_weight_bits": Q_BITS,
        "state_bits_by_boundary": list(STATE_BITS_BY_BOUNDARY),
        "signed_requantization": "nearest; exact half away from zero; integer division; no signed shift",
        "cotangent": (
            "bilinear-down(real uint8 0.mkv frame,874x1164->384x512); "
            "clip(rint(value-127.5),-127,127)"
        ),
        "chain": [str(plan["name"]) for plan in build_full_r_plans()],
    }


def _base_full_receipt(
    *, gt_cache: Path, pair_start: int, pair_count: int, n: int
) -> dict[str, Any]:
    if not gt_cache.is_file():
        raise FileNotFoundError(gt_cache)
    cache_sha = _sha256_file(gt_cache)
    contract = _full_contract_payload(
        gt_cache=gt_cache,
        cache_sha256=cache_sha,
        pair_start=pair_start,
        pair_count=pair_count,
        n=n,
    )
    contract_fingerprint = _sha256_bytes(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return {
        "schema": FULL_R_SCHEMA,
        "lane_id": "throughput_authority_ladder",
        "task_id": 494,
        "axis": "[macOS-MLX research-signal; NumPy-fp32/int32 authority; non-promotable MEANS]",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "training": False,
        "paid_dispatch": False,
        "live_run_mutation": False,
        "contract": contract,
        "contract_fingerprint": contract_fingerprint,
        "git_head_at_probe": _git_head(),
        "host": platform.node(),
        "platform": platform.platform(),
        "source_custody": {
            "probe": {
                "path": "tools/probe_pythagorean_exact_arithmetic_bitident.py",
                "sha256": _sha256_file(Path(__file__)),
            },
            "gt_cache": {"path": str(gt_cache), "sha256": cache_sha},
            "fused_r_reference": {
                "path": "src/tac/local_acceleration/metal_fused_r_operator.py",
                "sha256": _sha256_file(
                    REPO / "src/tac/local_acceleration/metal_fused_r_operator.py"
                ),
            },
        },
        "static_integer_proof": full_r_static_overflow_proof(),
        "derived_integer_error_bound": derive_full_r_integer_error_bound(),
        "precision_manifest": full_r_precision_manifest(),
        "numpy_authority": {"rows": []},
        "trials": {variant: [] for variant in VARIANTS},
    }


def _full_authority_summary(receipt: dict[str, Any]) -> dict[str, Any]:
    rows = list(receipt["numpy_authority"].get("rows", []))
    contract = receipt["contract"]
    expected_frames = int(contract["frames"])
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["pair_index"]),
            FULL_R_MEMBERS.index(str(row["member"])),
        ),
    )
    float_digest = hashlib.sha256()
    integer_digest = hashlib.sha256()
    for row in ordered:
        _update_corpus_digest(
            float_digest,
            pair_index=int(row["pair_index"]),
            member=str(row["member"]),
            output_sha256=str(row["float_output_sha256"]),
        )
        _update_corpus_digest(
            integer_digest,
            pair_index=int(row["pair_index"]),
            member=str(row["member"]),
            output_sha256=str(row["integer_output_sha256"]),
        )
    elements = sum(int(row["error_elements"]) for row in ordered)
    sum_squared = sum(float(row["sum_squared_error"]) for row in ordered)
    maximum = max((float(row["max_abs_error"]) for row in ordered), default=0.0)
    bound = float(receipt["derived_integer_error_bound"]["final_max_abs_error_bound"])
    keys = {(int(row["pair_index"]), str(row["member"])) for row in ordered}
    expected_keys = {
        (pair_index, member)
        for pair_index in range(
            int(contract["pair_start"]),
            int(contract["pair_start"]) + int(contract["pair_count"]),
        )
        for member in FULL_R_MEMBERS
    }
    return {
        "status": "MEASURED" if len(ordered) == expected_frames else "INCOMPLETE",
        "frames": len(ordered),
        "expected_frames": expected_frames,
        "coverage_exact": keys == expected_keys,
        "float_corpus_sha256": float_digest.hexdigest() if ordered else None,
        "integer_corpus_sha256": integer_digest.hexdigest() if ordered else None,
        "dequantized_max_abs_error_vs_numpy_fp32": maximum,
        "dequantized_rmse_vs_numpy_fp32": (
            float(np.sqrt(sum_squared / elements)) if elements else None
        ),
        "derived_max_abs_error_bound": bound,
        "within_derived_bound": bool(len(ordered) == expected_frames and maximum <= bound),
    }


def _ensure_full_numpy_authority(receipt: dict[str, Any], *, output: Path) -> None:
    rows: list[dict[str, Any]] = receipt["numpy_authority"].setdefault("rows", [])
    complete_keys = {(int(row["pair_index"]), str(row["member"])) for row in rows}
    contract = receipt["contract"]
    plans = build_full_r_plans()
    final_state_scale = float(1 << STATE_BITS_BY_BOUNDARY[-1])
    gt_cache = Path(contract["gt_cache"])
    for pair_index, member, frame in _ordered_real_frames(
        gt_cache=gt_cache,
        pair_start=int(contract["pair_start"]),
        pair_count=int(contract["pair_count"]),
    ):
        if (pair_index, member) in complete_keys:
            continue
        fixture = prepare_real_full_r_fixture(frame)
        float_output = full_r_vjp_numpy_float(
            fixture["cotangent"], fixture["clip_mask"], plans=plans
        )
        integer_output, proofs = full_r_vjp_numpy_integer(
            fixture["cotangent"], fixture["clip_mask"], plans=plans
        )
        dequantized = integer_output.astype(np.float64) / final_state_scale
        difference = dequantized - float_output.astype(np.float64)
        row = {
            "pair_index": int(pair_index),
            "member": member,
            "input_frame_sha256": _hash_array(frame),
            "cotangent_sha256": _hash_array(fixture["cotangent"]),
            "clip_mask_sha256": _hash_array(fixture["clip_mask"]),
            "float_output_sha256": _hash_array(float_output),
            "integer_output_sha256": _hash_array(integer_output),
            "max_abs_error": float(np.max(np.abs(difference))),
            "sum_squared_error": float(np.sum(difference * difference, dtype=np.float64)),
            "error_elements": int(difference.size),
            "stage_actual_max_sum_abs_contributions": [
                int(proof["actual_max_sum_abs_contributions"]) for proof in proofs
            ],
            "stage_actual_minimum_signed_accumulator_bits": [
                int(proof["actual_minimum_signed_accumulator_bits"]) for proof in proofs
            ],
        }
        rows.append(row)
        complete_keys.add((pair_index, member))
        receipt["numpy_authority"]["summary"] = _full_authority_summary(receipt)
        _atomic_write_json(output, receipt)
    receipt["numpy_authority"]["summary"] = _full_authority_summary(receipt)
    _atomic_write_json(output, receipt)


def _summarize_full(receipt: dict[str, Any]) -> dict[str, Any]:
    authority = receipt.get("numpy_authority", {}).get("summary", {})
    expected_n = int(receipt["contract"]["n_processes_per_variant"])
    expected_frames = int(receipt["contract"]["frames"])
    variants: dict[str, Any] = {}
    blocked = False
    for variant in VARIANTS:
        rows = list(receipt.get("trials", {}).get(variant, []))
        measured = [row for row in rows if row.get("status") == "MEASURED"]
        blockers = [row for row in rows if row.get("status") == "BLOCKED_NOT_MEASURED"]
        hashes = [str(row["corpus_sha256"]) for row in measured]
        entry = {
            "attempts": len(rows),
            "measured_processes": len(measured),
            "expected_processes": expected_n,
            "all_full_coverage": bool(
                measured
                and all(int(row.get("frames", -1)) == expected_frames for row in measured)
            ),
            "unique_corpus_hashes": len(set(hashes)),
            "cross_process_identical": bool(hashes and len(set(hashes)) == 1),
            "hashes": hashes,
            "blockers": blockers,
        }
        if variant == "fixed_q15_int32_atomic":
            authority_hash = authority.get("integer_corpus_sha256")
            entry["exact_numpy_int_corpus_parity"] = bool(
                hashes and authority_hash and all(value == authority_hash for value in hashes)
            )
        variants[variant] = entry
        blocked = blocked or bool(blockers)
    complete = bool(
        authority.get("status") == "MEASURED"
        and authority.get("coverage_exact")
        and all(variants[variant]["measured_processes"] == expected_n for variant in VARIANTS)
    )
    float_diverges = bool(
        complete and variants["float_atomic"]["unique_corpus_hashes"] > 1
    )
    integer_holds = bool(
        complete
        and variants["fixed_q15_int32_atomic"]["cross_process_identical"]
        and variants["fixed_q15_int32_atomic"]["exact_numpy_int_corpus_parity"]
        and authority.get("within_derived_bound")
    )
    if blocked:
        verdict = "BLOCKED_NOT_MEASURED"
        verdict_scope = "ENVIRONMENT: no evaluated Metal device in attempted child process"
    elif not complete:
        verdict = "INCOMPLETE"
        verdict_scope = "INSTANCE: full-R real-n600 receipt coverage/process count"
    elif float_diverges and integer_holds:
        verdict = "REAL-L70-LEVER-FULL-R-N600"
        verdict_scope = (
            "n600 INSTANCE: real 0.mkv gt_f0+gt_f1, four-axis render-R VJP, "
            "Q15/int32 atomic with Q7/Q5 state schedule, this MLX/Metal host"
        )
    elif not integer_holds:
        verdict = "FULL-R-INTEGER-FORMULATION-NO-GO"
        verdict_scope = (
            "FORMULATION: Q15/int32 atomic plus Q7/Q5 boundary schedule; not the integer family"
        )
    else:
        verdict = "FLOAT-WALL-NOT-REPRODUCED-FULL-R"
        verdict_scope = "INSTANCE: this full-R real-n600 corpus/host/process set"
    return {
        **variants,
        "authority": authority,
        "complete": complete,
        "decisive_positive": bool(float_diverges and integer_holds),
        "overall_verdict": verdict,
        "verdict_scope": verdict_scope,
    }


def run_full_parent(
    *, output: Path, gt_cache: Path, pair_start: int, pair_count: int, n: int, resume: bool
) -> dict[str, Any]:
    if n < 2:
        raise ValueError("n must be >=2 for a cross-process verdict")
    if resume and output.exists():
        receipt = json.loads(output.read_text(encoding="utf-8"))
        if receipt.get("schema") != FULL_R_SCHEMA:
            raise ValueError(f"cannot resume incompatible full-R receipt {output}")
        current_sha = _sha256_file(Path(__file__))
        recorded_sha = receipt.get("source_custody", {}).get("probe", {}).get("sha256")
        if current_sha != recorded_sha:
            raise ValueError(
                "resume probe bytes differ from full-R receipt custody; start a fresh receipt "
                f"({recorded_sha!r} != {current_sha!r})"
            )
        expected = receipt["contract"]
        requested = _full_contract_payload(
            gt_cache=gt_cache,
            cache_sha256=str(expected["gt_cache_sha256"]),
            pair_start=pair_start,
            pair_count=pair_count,
            n=n,
        )
        requested_fingerprint = _sha256_bytes(
            json.dumps(requested, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        if requested_fingerprint != receipt.get("contract_fingerprint"):
            raise ValueError("resume full-R contract fingerprint differs")
    else:
        receipt = _base_full_receipt(
            gt_cache=gt_cache,
            pair_start=pair_start,
            pair_count=pair_count,
            n=n,
        )
        _atomic_write_json(output, receipt)
    _ensure_full_numpy_authority(receipt, output=output)
    if receipt["numpy_authority"]["summary"]["status"] != "MEASURED":
        receipt["summary"] = _summarize_full(receipt)
        _atomic_write_json(output, receipt)
        return receipt
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    for variant in VARIANTS:
        rows: list[dict[str, Any]] = receipt["trials"][variant]
        if any(row.get("status") == "BLOCKED_NOT_MEASURED" for row in rows):
            continue
        while len([row for row in rows if row.get("status") == "MEASURED"]) < n:
            trial_index = len(rows)
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--scope",
                    FULL_R_SCOPE,
                    "--child-full",
                    variant,
                    "--gt-cache",
                    str(gt_cache),
                    "--pair-start",
                    str(pair_start),
                    "--pair-count",
                    str(pair_count),
                ],
                cwd=REPO,
                env=env,
                capture_output=True,
                text=True,
                timeout=7200,
                check=False,
            )
            if process.returncode != 0:
                receipt["failure"] = {
                    "variant": variant,
                    "trial_index": trial_index,
                    "returncode": process.returncode,
                    "stderr_tail": process.stderr[-2400:],
                    "stdout_tail": process.stdout[-1200:],
                }
                receipt["summary"] = _summarize_full(receipt)
                _atomic_write_json(output, receipt)
                raise RuntimeError(
                    f"full-R child failed for {variant} trial {trial_index}: "
                    f"{process.stderr[-400:]}"
                )
            row = json.loads(process.stdout.strip().splitlines()[-1])
            row["trial_index"] = trial_index
            rows.append(row)
            receipt["summary"] = _summarize_full(receipt)
            _atomic_write_json(output, receipt)
            if row.get("status") == "BLOCKED_NOT_MEASURED":
                break
    receipt["summary"] = _summarize_full(receipt)
    receipt["completed"] = bool(receipt["summary"]["complete"])
    _atomic_write_json(output, receipt)
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("one-axis", FULL_R_SCOPE), default="one-axis")
    parser.add_argument("--child", choices=VARIANTS, help="internal: execute one MLX process")
    parser.add_argument("--child-full", choices=VARIANTS, help="internal: execute one full-R process")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n", type=int, default=N_PROCESSES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--numpy-only", action="store_true", help="emit the pure NumPy fixture contract")
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_GT_CACHE)
    parser.add_argument("--pair-start", type=int, default=0)
    parser.add_argument("--pair-count", type=int, default=FULL_R_PAIR_COUNT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.child and args.child_full:
        raise ValueError("--child and --child-full are mutually exclusive")
    if args.child:
        print(json.dumps(run_mlx_variant(args.child), sort_keys=True))
        return 0
    if args.child_full:
        print(
            json.dumps(
                run_full_mlx_variant(
                    variant=args.child_full,
                    gt_cache=args.gt_cache,
                    pair_start=args.pair_start,
                    pair_count=args.pair_count,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.numpy_only:
        if args.scope == FULL_R_SCOPE:
            plans = build_full_r_plans()
            print(
                json.dumps(
                    {
                        "scope": FULL_R_SCOPE,
                        "chain": [str(plan["name"]) for plan in plans],
                        "state_bits_by_boundary": list(STATE_BITS_BY_BOUNDARY),
                        "static_integer_proof": full_r_static_overflow_proof(plans),
                        "derived_integer_error_bound": derive_full_r_integer_error_bound(plans),
                        "precision_manifest": full_r_precision_manifest(plans),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        fixture = build_resize_adjoint_fixture()
        print(
            json.dumps(
                {
                    "in_size": IN_SIZE,
                    "out_size": OUT_SIZE,
                    "lanes": LANES,
                    "q_bits": Q_BITS,
                    "max_abs_integer_accumulator": fixture["max_abs_integer_accumulator"],
                    "max_contributions_per_destination": fixture["max_contributions_per_destination"],
                    "authority_tolerance": fixture["authority_tolerance"],
                    "float_reorder_tolerance": fixture["float_reorder_tolerance"],
                    "int_reference_sha256": _hash_array(np.asarray(fixture["int_reference"], dtype=np.int32)),
                    "float_reference_sha256": _hash_array(np.asarray(fixture["float_reference"], dtype=np.float32)),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.scope == FULL_R_SCOPE:
        output = args.output
        if output == DEFAULT_OUTPUT:
            output = DEFAULT_FULL_R_OUTPUT
        receipt = run_full_parent(
            output=output,
            gt_cache=args.gt_cache,
            pair_start=args.pair_start,
            pair_count=args.pair_count,
            n=args.n,
            resume=args.resume,
        )
        print(json.dumps(receipt["summary"], indent=2, sort_keys=True))
        return 0
    receipt = run_parent(output=args.output, n=args.n, resume=args.resume)
    print(json.dumps(receipt["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
