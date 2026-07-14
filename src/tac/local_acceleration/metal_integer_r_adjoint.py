# SPDX-License-Identifier: MIT
"""Order-independent fixed-point Metal adjoint for the contest render-R.

This is a default-OFF training-gradient backend, not a score authority.  It
keeps the existing contest-faithful float forward and replaces only the VJP's
four resize transposes with Q15-weight/int32 gather kernels.  Integer addition
is associative while the registered overflow proof holds, so the result does
not depend on a floating reduction order.  One thread owns one output element;
there are no atomics and no cross-thread write collisions.

Promotion is fail closed: a caller must supply a typed, receipt-admitted policy.
The frozen NumPy-fp32 scorer and exact contest CPU/CUDA replay remain terminal
score authority even after this backend is admitted for training.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

INTEGER_R_ADJOINT_FLAG = "TAC_MLX_INTEGER_R_ADJOINT"
Q_WEIGHT_BITS = 15
Q_WEIGHT_SCALE = 1 << Q_WEIGHT_BITS
STATE_BITS_BY_BOUNDARY = (7, 7, 7, 5, 5)
INT32_LIMIT = int(np.iinfo(np.int32).max)
_TRUTHY = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class IntegerTransposePlan:
    """CSR plan for one forward-resize transpose."""

    name: str
    in_size: int
    out_size: int
    mode: str
    starts: np.ndarray
    counts: np.ndarray
    source_indices: np.ndarray
    q15_weights: np.ndarray
    max_fan_in: int
    max_q15_abs_weight_sum: int
    max_float_abs_weight_sum: float
    max_q15_l1_error: float


@dataclass(frozen=True)
class IntegerRAdjointConfig:
    """Typed opt-in parameters for the approximate training VJP."""

    cotangent_unit: float
    maximum_abs_cotangent: float = 127.0
    q_weight_bits: int = Q_WEIGHT_BITS
    state_bits_by_boundary: tuple[int, ...] = STATE_BITS_BY_BOUNDARY

    def validate(self) -> None:
        if not np.isfinite(self.cotangent_unit) or self.cotangent_unit <= 0.0:
            raise ValueError("cotangent_unit must be finite and positive")
        if not np.isfinite(self.maximum_abs_cotangent) or self.maximum_abs_cotangent <= 0.0:
            raise ValueError("maximum_abs_cotangent must be finite and positive")
        if int(self.q_weight_bits) != Q_WEIGHT_BITS:
            raise ValueError(f"only Q{Q_WEIGHT_BITS} weights are registered")
        if tuple(self.state_bits_by_boundary) != STATE_BITS_BY_BOUNDARY:
            raise ValueError(
                f"unregistered state schedule {self.state_bits_by_boundary}; "
                f"expected {STATE_BITS_BY_BOUNDARY}"
            )


def _cubic_weight(distance: np.ndarray) -> np.ndarray:
    x = np.abs(distance).astype(np.float32)
    a = np.float32(-0.75)
    x2 = (x * x).astype(np.float32)
    x3 = (x2 * x).astype(np.float32)
    inner = (
        (a + np.float32(2.0)) * x3
        - (a + np.float32(3.0)) * x2
        + np.float32(1.0)
    ).astype(np.float32)
    outer = (
        a * x3
        - np.float32(5.0) * a * x2
        + np.float32(8.0) * a * x
        - np.float32(4.0) * a
    ).astype(np.float32)
    return np.where(
        x <= np.float32(1.0),
        inner,
        np.where(x < np.float32(2.0), outer, np.float32(0.0)),
    ).astype(np.float32)


def resize_taps(*, in_size: int, out_size: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Mirror render-R align-corners-false bilinear/a=-0.75 bicubic taps."""

    in_size = int(in_size)
    out_size = int(out_size)
    scale = np.float32(float(in_size) / float(out_size))
    output = np.arange(out_size, dtype=np.float32)
    real = ((output + np.float32(0.5)) * scale - np.float32(0.5)).astype(np.float32)
    base = np.floor(real).astype(np.int32)
    normalized = str(mode).strip().lower()
    if normalized == "bilinear":
        right = (real - base.astype(np.float32)).astype(np.float32)
        indices = np.stack([base, base + 1], axis=1)
        weights = np.stack([np.float32(1.0) - right, right], axis=1).astype(np.float32)
    elif normalized == "bicubic":
        offsets = np.asarray([-1, 0, 1, 2], dtype=np.int32)
        unclipped = base[:, None] + offsets[None, :]
        indices = unclipped
        weights = _cubic_weight(real[:, None] - unclipped.astype(np.float32))
    else:
        raise ValueError(f"unsupported resize mode {mode!r}")
    return np.clip(indices, 0, in_size - 1).astype(np.int32), weights


def build_integer_transpose_plan(
    *, name: str, in_size: int, out_size: int, mode: str
) -> IntegerTransposePlan:
    """Quantize original forward taps and retain duplicates in stable CSR order."""

    indices, weights = resize_taps(in_size=in_size, out_size=out_size, mode=mode)
    qweights = np.rint(weights.astype(np.float64) * float(Q_WEIGHT_SCALE)).astype(np.int32)
    rows: list[list[tuple[int, int, np.float32]]] = [[] for _ in range(int(in_size))]
    for output_index in range(int(out_size)):
        for tap_index in range(int(indices.shape[1])):
            rows[int(indices[output_index, tap_index])].append(
                (
                    output_index,
                    int(qweights[output_index, tap_index]),
                    np.float32(weights[output_index, tap_index]),
                )
            )
    starts = np.zeros((int(in_size),), dtype=np.int32)
    counts = np.zeros((int(in_size),), dtype=np.int32)
    source: list[int] = []
    qflat: list[int] = []
    maximum_qsum = 0
    maximum_float_sum = 0.0
    maximum_error = 0.0
    for destination, row in enumerate(rows):
        starts[destination] = len(source)
        counts[destination] = len(row)
        qsum = 0
        float_sum = 0.0
        error_sum = 0.0
        for output_index, qvalue, float_value in row:
            source.append(output_index)
            qflat.append(qvalue)
            qsum += abs(qvalue)
            float_sum += abs(float(float_value))
            error_sum += abs(float(qvalue) / float(Q_WEIGHT_SCALE) - float(float_value))
        maximum_qsum = max(maximum_qsum, qsum)
        maximum_float_sum = max(maximum_float_sum, float_sum)
        maximum_error = max(maximum_error, error_sum)
    return IntegerTransposePlan(
        name=str(name),
        in_size=int(in_size),
        out_size=int(out_size),
        mode=str(mode),
        starts=starts,
        counts=counts,
        source_indices=np.asarray(source, dtype=np.int32),
        q15_weights=np.asarray(qflat, dtype=np.int32),
        max_fan_in=int(np.max(counts)) if counts.size else 0,
        max_q15_abs_weight_sum=int(maximum_qsum),
        max_float_abs_weight_sum=float(maximum_float_sum),
        max_q15_l1_error=float(maximum_error),
    )


@lru_cache(maxsize=1)
def full_r_integer_plans() -> tuple[IntegerTransposePlan, ...]:
    return (
        build_integer_transpose_plan(
            name="down_w_transpose_512_to_1164", in_size=1164, out_size=512, mode="bilinear"
        ),
        build_integer_transpose_plan(
            name="down_h_transpose_384_to_874", in_size=874, out_size=384, mode="bilinear"
        ),
        build_integer_transpose_plan(
            name="up_w_transpose_1164_to_512", in_size=512, out_size=1164, mode="bicubic"
        ),
        build_integer_transpose_plan(
            name="up_h_transpose_874_to_384", in_size=384, out_size=874, mode="bicubic"
        ),
    )


def signed_round_divide(values: Any, divisor: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.int64)
    divisor = int(divisor)
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    magnitude = np.abs(source)
    rounded = (magnitude + divisor // 2) // divisor
    output = np.where(source < 0, -rounded, rounded)
    if output.size and (
        int(np.min(output)) < np.iinfo(np.int32).min
        or int(np.max(output)) > np.iinfo(np.int32).max
    ):
        raise OverflowError("requantized integer R-adjoint output exceeds int32")
    return output.astype(np.int32)


def minimum_signed_bits_for_bound(bound: int) -> int:
    bound = int(bound)
    if bound < 0:
        raise ValueError("bound must be non-negative")
    return max(1, int(np.ceil(np.log2(2 * bound + 1)))) if bound else 1


def prove_stage_int32(
    plan: IntegerTransposePlan,
    *,
    max_abs_input_integer: int,
    in_state_bits: int,
    out_state_bits: int,
) -> dict[str, Any]:
    bound = int(max_abs_input_integer) * int(plan.max_q15_abs_weight_sum)
    shift = Q_WEIGHT_BITS + int(in_state_bits) - int(out_state_bits)
    if shift < 0:
        raise ValueError("registered integer R-adjoint never left-shifts")
    divisor = 1 << shift
    safe_limit = INT32_LIMIT - divisor // 2
    safe = bound <= safe_limit
    result = {
        "stage": plan.name,
        "bound_kind": "STATIC_WORST_CASE_LINF",
        "max_fan_in": plan.max_fan_in,
        "max_abs_input_integer": int(max_abs_input_integer),
        "max_q15_abs_weight_sum": plan.max_q15_abs_weight_sum,
        "max_abs_accumulator_bound": int(bound),
        "minimum_signed_accumulator_bits": minimum_signed_bits_for_bound(bound),
        "implemented_accumulator_bits": 32,
        "safe_limit": int(safe_limit),
        "headroom_x": float(safe_limit) / float(bound) if bound else float("inf"),
        "requant_divisor": int(divisor),
        "in_state_bits": int(in_state_bits),
        "out_state_bits": int(out_state_bits),
        "max_abs_output_integer_bound": int((bound + divisor // 2) // divisor),
        "safe": bool(safe),
        "reorder_identity_preconditions": (
            "exact integer add over every reachable partial sum; no overflow or saturation; "
            "one deterministic signed requantization"
        ),
    }
    if not safe:
        raise OverflowError(f"{plan.name} bound {bound} exceeds int32 safe limit {safe_limit}")
    return result


def full_r_int32_proof(*, maximum_abs_cotangent: float = 127.0) -> list[dict[str, Any]]:
    if not np.isfinite(maximum_abs_cotangent) or maximum_abs_cotangent <= 0.0:
        raise ValueError("maximum_abs_cotangent must be finite and positive")
    maximum = int(np.ceil(maximum_abs_cotangent * (1 << STATE_BITS_BY_BOUNDARY[0])))
    rows: list[dict[str, Any]] = []
    for index, plan in enumerate(full_r_integer_plans()):
        row = prove_stage_int32(
            plan,
            max_abs_input_integer=maximum,
            in_state_bits=STATE_BITS_BY_BOUNDARY[index],
            out_state_bits=STATE_BITS_BY_BOUNDARY[index + 1],
        )
        rows.append(row)
        maximum = int(row["max_abs_output_integer_bound"])
    return rows


def integer_transpose_numpy(
    input_l_sout_r: Any,
    plan: IntegerTransposePlan,
    *,
    in_state_bits: int,
    out_state_bits: int,
) -> np.ndarray:
    source = np.asarray(input_l_sout_r, dtype=np.int32)
    if source.ndim != 3 or int(source.shape[1]) != plan.out_size:
        raise ValueError(f"{plan.name} expected (L,{plan.out_size},R), got {source.shape}")
    prove_stage_int32(
        plan,
        max_abs_input_integer=(
            int(np.max(np.abs(source.astype(np.int64)))) if source.size else 0
        ),
        in_state_bits=in_state_bits,
        out_state_bits=out_state_bits,
    )
    output = np.zeros((source.shape[0], plan.in_size, source.shape[2]), dtype=np.int64)
    for destination in range(plan.in_size):
        start = int(plan.starts[destination])
        count = int(plan.counts[destination])
        for offset in range(count):
            slot = start + offset
            output[:, destination, :] += (
                source[:, int(plan.source_indices[slot]), :].astype(np.int64)
                * np.int64(plan.q15_weights[slot])
            )
    shift = Q_WEIGHT_BITS + int(in_state_bits) - int(out_state_bits)
    return signed_round_divide(output, 1 << shift)


_INTEGER_TRANSPOSE_KERNEL: Any | None = None
_MX_PLAN_CACHE: dict[str, tuple[Any, Any, Any, Any]] = {}
_KERNEL_LOCK = threading.Lock()


def _kernel() -> Any:
    global _INTEGER_TRANSPOSE_KERNEL
    with _KERNEL_LOCK:
        if _INTEGER_TRANSPOSE_KERNEL is None:
            import mlx.core as mx

            _INTEGER_TRANSPOSE_KERNEL = mx.fast.metal_kernel(
                name="integer_r_transpose_q15_i32_gather",
                input_names=["inp", "starts", "counts", "source", "weight", "dims"],
                output_names=["out"],
                source="""
                    uint gid = thread_position_in_grid.x;
                    int L = dims[0];
                    int Sout = dims[1];
                    int Sin = dims[2];
                    int R = dims[3];
                    int divisor = dims[4];
                    int total = L * Sin * R;
                    if (gid >= (uint)total) return;
                    int r = int(gid) % R;
                    int destination = (int(gid) / R) % Sin;
                    int left = int(gid) / (R * Sin);
                    int start = starts[destination];
                    int count = counts[destination];
                    int accumulator = 0;
                    for (int offset = 0; offset < count; ++offset) {
                        int slot = start + offset;
                        int src = (left * Sout + source[slot]) * R + r;
                        accumulator += inp[src] * weight[slot];
                    }
                    int magnitude = accumulator < 0 ? -accumulator : accumulator;
                    int rounded = (magnitude + divisor / 2) / divisor;
                    out[gid] = accumulator < 0 ? -rounded : rounded;
                """,
            )
    return _INTEGER_TRANSPOSE_KERNEL


def metal_integer_r_available() -> bool:
    try:
        import mlx.core as mx

        if mx.default_device().type != mx.gpu:
            return False
        probe = mx.sum(mx.arange(8, dtype=mx.float32))
        mx.eval(probe)
        return float(probe.item()) == 28.0
    except Exception:
        return False


def _mx_plan(plan: IntegerTransposePlan) -> tuple[Any, Any, Any, Any]:
    import mlx.core as mx

    cached = _MX_PLAN_CACHE.get(plan.name)
    if cached is None:
        cached = (
            mx.array(plan.starts, dtype=mx.int32),
            mx.array(plan.counts, dtype=mx.int32),
            mx.array(plan.source_indices, dtype=mx.int32),
            mx.array(plan.q15_weights, dtype=mx.int32),
        )
        _MX_PLAN_CACHE[plan.name] = cached
    return cached


def integer_transpose_metal(
    input_l_sout_r: Any,
    plan: IntegerTransposePlan,
    *,
    in_state_bits: int,
    out_state_bits: int,
) -> Any:
    import mlx.core as mx

    if not metal_integer_r_available():
        raise RuntimeError("integer R-adjoint requested without an evaluated Metal device")
    left, sout, right = map(int, input_l_sout_r.shape)
    if sout != plan.out_size:
        raise ValueError(f"{plan.name} expected Sout={plan.out_size}, got {sout}")
    starts, counts, source, weights = _mx_plan(plan)
    shift = Q_WEIGHT_BITS + int(in_state_bits) - int(out_state_bits)
    divisor = 1 << shift
    dims = mx.array([left, sout, plan.in_size, right, divisor], dtype=mx.int32)
    total = left * plan.in_size * right
    (output,) = _kernel()(
        inputs=[input_l_sout_r, starts, counts, source, weights, dims],
        output_shapes=[(left, plan.in_size, right)],
        output_dtypes=[mx.int32],
        grid=(total, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return output


def integer_r_vjp_state_metal(
    cotangent_nhwc: Any,
    clip_mask_nhwc: Any,
    *,
    config: IntegerRAdjointConfig,
) -> Any:
    """Apply four integer gather kernels and return the final exact int32 state."""

    import mlx.core as mx

    config.validate()
    full_r_int32_proof(maximum_abs_cotangent=config.maximum_abs_cotangent)
    if tuple(map(int, cotangent_nhwc.shape[-3:])) != (384, 512, 3):
        raise ValueError(f"expected (...,384,512,3) cotangent, got {cotangent_nhwc.shape}")
    lead = tuple(map(int, cotangent_nhwc.shape[:-3]))
    batch = int(np.prod(lead, dtype=np.int64)) if lead else 1
    scale = float(1 << STATE_BITS_BY_BOUNDARY[0]) / float(config.cotangent_unit)
    bounded = mx.clip(
        cotangent_nhwc,
        -float(config.maximum_abs_cotangent) * float(config.cotangent_unit),
        float(config.maximum_abs_cotangent) * float(config.cotangent_unit),
    )
    value = mx.round(bounded * scale).astype(mx.int32)
    plans = full_r_integer_plans()
    value = integer_transpose_metal(
        mx.reshape(value, (batch * 384, 512, 3)),
        plans[0],
        in_state_bits=STATE_BITS_BY_BOUNDARY[0],
        out_state_bits=STATE_BITS_BY_BOUNDARY[1],
    )
    value = integer_transpose_metal(
        mx.reshape(value, (batch, 384, 1164 * 3)),
        plans[1],
        in_state_bits=STATE_BITS_BY_BOUNDARY[1],
        out_state_bits=STATE_BITS_BY_BOUNDARY[2],
    )
    value = mx.reshape(value, (batch, 874, 1164, 3))
    value = value * mx.reshape(clip_mask_nhwc.astype(mx.int32), (batch, 874, 1164, 3))
    value = integer_transpose_metal(
        mx.reshape(value, (batch * 874, 1164, 3)),
        plans[2],
        in_state_bits=STATE_BITS_BY_BOUNDARY[2],
        out_state_bits=STATE_BITS_BY_BOUNDARY[3],
    )
    value = integer_transpose_metal(
        mx.reshape(value, (batch, 874, 512 * 3)),
        plans[3],
        in_state_bits=STATE_BITS_BY_BOUNDARY[3],
        out_state_bits=STATE_BITS_BY_BOUNDARY[4],
    )
    return mx.reshape(value, (*lead, 384, 512, 3))


def integer_r_vjp_metal(
    cotangent_nhwc: Any,
    clip_mask_nhwc: Any,
    *,
    config: IntegerRAdjointConfig,
) -> Any:
    """Apply four integer gather kernels and return the dequantized fp32 gradient."""

    import mlx.core as mx

    value = integer_r_vjp_state_metal(cotangent_nhwc, clip_mask_nhwc, config=config)
    output_scale = float(config.cotangent_unit) / float(1 << STATE_BITS_BY_BOUNDARY[-1])
    return value.astype(mx.float32) * output_scale


def make_integer_r_roundtrip(*, config: IntegerRAdjointConfig) -> Any:
    """Existing exact float forward plus the receipt-gated integer VJP."""

    import mlx.core as mx

    from tac.local_acceleration.metal_fused_r_operator import (
        _fused_r_metal_forward,
        _resample_mx,
    )

    config.validate()
    full_r_int32_proof(maximum_abs_cotangent=config.maximum_abs_cotangent)

    @mx.custom_function
    def roundtrip(x: Any) -> Any:
        return _fused_r_metal_forward(
            x, camera_hw=(874, 1164), output_hw=(384, 512), ste_round=True
        )

    @roundtrip.vjp
    def _vjp(primals: Any, cotangent: Any, output: Any) -> tuple[Any]:
        del output
        x = primals[0] if isinstance(primals, (tuple, list)) else primals
        cot = cotangent[0] if isinstance(cotangent, (tuple, list)) else cotangent
        lead = tuple(map(int, x.shape[:-3]))
        batch = int(np.prod(lead, dtype=np.int64)) if lead else 1
        flat = mx.reshape(x, (batch, 384, 512, 3))
        camera = _resample_mx(
            flat, out_h=874, out_w=1164, mode="bicubic", do_round=False
        )
        mask = mx.logical_and(camera > 0.0, camera < 255.0)
        gradient = integer_r_vjp_metal(cot, mask, config=config)
        return (gradient,)

    return roundtrip


def integer_r_env_requested() -> bool:
    return os.environ.get(INTEGER_R_ADJOINT_FLAG, "").strip().lower() in _TRUTHY


def receipt_admits_integer_r(
    receipt_path: str | Path,
    *,
    expected_probe_sha256: str,
    expected_policy_sha256: str,
) -> dict[str, Any]:
    """Validate the full-n600 Metal receipt and bind it to current source bytes."""

    path = Path(receipt_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    summary = receipt.get("summary", {})
    source = receipt.get("source_custody", {})
    failures: list[str] = []
    if receipt.get("schema") != "pythagorean_exact_arithmetic_full_r_n600.v2":
        failures.append("schema")
    if summary.get("overall_verdict") != "REAL-L70-LEVER-FULL-R-N600":
        failures.append("verdict")
    if not summary.get("complete") or not summary.get("decisive_positive"):
        failures.append("coverage")
    if source.get("probe", {}).get("sha256") != expected_probe_sha256:
        failures.append("probe_sha256")
    if receipt.get("training_integration", {}).get("policy_sha256") != expected_policy_sha256:
        failures.append("policy_sha256")
    benchmark = receipt.get("training_integration", {}).get("kernel_benchmark", {})
    if not benchmark.get("measured") or float(benchmark.get("speedup_x", 0.0)) <= 1.0:
        failures.append("positive_speed")
    if failures:
        raise ValueError(f"integer R-adjoint receipt is not admissible: {failures}")
    return receipt


def integer_r_signature() -> dict[str, Any]:
    plans = full_r_integer_plans()
    proof = full_r_int32_proof()
    source = _kernel.__module__.encode("utf-8") + json.dumps(
        [plan.name for plan in plans], sort_keys=True
    ).encode("utf-8")
    return {
        "schema": "metal_integer_r_adjoint_signature.v1",
        "built": True,
        "default_enabled": False,
        "env_flag": INTEGER_R_ADJOINT_FLAG,
        "research_only": True,
        "score_claim": False,
        "operation": "four-axis render-R transpose VJP",
        "arithmetic": "Q15 weights; int32 gather accumulation; Q7/Q5 state",
        "atomic": False,
        "order_dependence": "none while int32 proof holds",
        "stage_names": [plan.name for plan in plans],
        "static_proof": proof,
        "precision_manifest": {
            "schema": "frontier_math_precision_manifest.v1",
            "bound_kind": "STATIC_WORST_CASE_LINF",
            "layers": [
                {
                    "op": row["stage"],
                    "options": [
                        {
                            "weight_bits": Q_WEIGHT_BITS,
                            "input_state_bits": row["in_state_bits"],
                            "output_state_bits": row["out_state_bits"],
                            "accumulator_bits": 32,
                            "minimum_exact_signed_accumulator_bits": row[
                                "minimum_signed_accumulator_bits"
                            ],
                            "measured_cost": None,
                            "measured_cost_status": "OWED_HOST_METAL",
                        }
                    ],
                }
                for row in proof
            ],
            "observed_corpus_is_not_unseen_input_ibp": True,
        },
        "source_semantic_sha256": hashlib.sha256(source).hexdigest(),
        "promotion_gate": "full-r-n600 N=10 receipt + NumPy-int parity + positive kernel speed",
        "terminal_authority": "unchanged exact contest CPU/CUDA archive replay",
        "verdict_scope": "training render-R VJP formulation only",
    }


__all__ = [
    "INT32_LIMIT",
    "INTEGER_R_ADJOINT_FLAG",
    "Q_WEIGHT_BITS",
    "STATE_BITS_BY_BOUNDARY",
    "IntegerRAdjointConfig",
    "IntegerTransposePlan",
    "build_integer_transpose_plan",
    "full_r_int32_proof",
    "full_r_integer_plans",
    "integer_r_env_requested",
    "integer_r_signature",
    "integer_r_vjp_metal",
    "integer_r_vjp_state_metal",
    "integer_transpose_metal",
    "integer_transpose_numpy",
    "make_integer_r_roundtrip",
    "metal_integer_r_available",
    "minimum_signed_bits_for_bound",
    "prove_stage_int32",
    "receipt_admits_integer_r",
    "resize_taps",
    "signed_round_divide",
]
