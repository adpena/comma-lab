# SPDX-License-Identifier: MIT
"""Compact-support and rank-batched frozen-convolution input adjoints on Metal.

This module is a compute primitive, not a sparse-SegNet accuracy claim.  A
caller supplies the output support (or a rank-r family of cotangents on that
support); the module computes only the input sites reached by that support.
It does not predict a support mask, approximate squeeze-excite state, or wire a
live trainer.  Frozen weights are treated as constants, so only the input VJP
is produced.

Numerical authority
-------------------
``dense_conv2d_input_adjoint_numpy_fp32`` is the fixed-order authority.  The
Metal kernel owns one ``(input-site, input-channel)`` per thread, gathers in
ascending ``kernel-row -> kernel-column -> output-channel`` order, disables
floating-point contraction, and uses no atomics.  Consequently a compact
result can be compared directly with the dense NumPy-fp32 result restricted to
the same propagated support.

Layouts
-------
* cotangent: ``(rank, active_output_sites, Cout)``
* weight: ``(Cout, kH, kW, Cin // groups)`` (MLX grouped-conv layout)
* result: ``(rank, active_input_sites, Cin)``

Rank is fused in registers up to :data:`MAX_FUSED_RANK`; larger bases are
handled by deterministic consecutive chunks.  This is the exact composition
needed to amortize basis VJPs across state-stable Jacobian reuse windows, but
the reuse/predictor still owns accuracy and admission.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

MAX_FUSED_RANK = 8
SPARSE_ADJOINT_KERNEL_NAME = "compact_sparse_conv2d_input_adjoint"

__all__ = [
    "MAX_FUSED_RANK",
    "SPARSE_ADJOINT_KERNEL_NAME",
    "Conv2DAdjointSpec",
    "MetalSparseAdjointExecutable",
    "SparseSpatialPlan",
    "build_sparse_spatial_plan",
    "compact_cotangent",
    "dense_conv2d_input_adjoint_numpy_fp32",
    "make_sparse_conv2d_input_adjoint_metal",
    "sparse_adjoint_metal_available",
    "sparse_conv2d_input_adjoint_metal",
    "sparse_conv2d_input_adjoint_numpy_fp32",
]


def _pair(value: int | Iterable[int], *, name: str) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} must contain exactly two integers")
        pair = int(value[0]), int(value[1])
    else:
        pair = int(value), int(value)
    return pair


@dataclass(frozen=True)
class Conv2DAdjointSpec:
    """Static geometry for a two-dimensional grouped-convolution input VJP."""

    input_hw: tuple[int, int]
    output_hw: tuple[int, int]
    cin: int
    cout: int
    kernel_hw: tuple[int, int]
    stride_hw: tuple[int, int] = (1, 1)
    padding_hw: tuple[int, int] = (0, 0)
    dilation_hw: tuple[int, int] = (1, 1)
    groups: int = 1

    def __post_init__(self) -> None:
        pairs = {
            "input_hw": self.input_hw,
            "output_hw": self.output_hw,
            "kernel_hw": self.kernel_hw,
            "stride_hw": self.stride_hw,
            "padding_hw": self.padding_hw,
            "dilation_hw": self.dilation_hw,
        }
        normalized = {name: _pair(value, name=name) for name, value in pairs.items()}
        for name, value in normalized.items():
            object.__setattr__(self, name, value)
        if min(self.input_hw) <= 0 or min(self.output_hw) <= 0:
            raise ValueError("input_hw and output_hw must be positive")
        if min(self.kernel_hw) <= 0 or min(self.stride_hw) <= 0 or min(self.dilation_hw) <= 0:
            raise ValueError("kernel, stride, and dilation must be positive")
        if min(self.padding_hw) < 0:
            raise ValueError("padding must be non-negative")
        if self.cin <= 0 or self.cout <= 0 or self.groups <= 0:
            raise ValueError("cin, cout, and groups must be positive")
        if self.cin % self.groups or self.cout % self.groups:
            raise ValueError("cin and cout must be divisible by groups")
        expected_h = (
            self.input_hw[0]
            + 2 * self.padding_hw[0]
            - self.dilation_hw[0] * (self.kernel_hw[0] - 1)
            - 1
        ) // self.stride_hw[0] + 1
        expected_w = (
            self.input_hw[1]
            + 2 * self.padding_hw[1]
            - self.dilation_hw[1] * (self.kernel_hw[1] - 1)
            - 1
        ) // self.stride_hw[1] + 1
        if self.output_hw != (expected_h, expected_w):
            raise ValueError(
                f"output_hw {self.output_hw} disagrees with convolution geometry "
                f"{(expected_h, expected_w)}"
            )

    @property
    def input_channels_per_group(self) -> int:
        return self.cin // self.groups

    @property
    def output_channels_per_group(self) -> int:
        return self.cout // self.groups

    @property
    def weight_shape(self) -> tuple[int, int, int, int]:
        return (
            self.cout,
            self.kernel_hw[0],
            self.kernel_hw[1],
            self.input_channels_per_group,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_hw": list(self.input_hw),
            "output_hw": list(self.output_hw),
            "cin": self.cin,
            "cout": self.cout,
            "kernel_hw": list(self.kernel_hw),
            "stride_hw": list(self.stride_hw),
            "padding_hw": list(self.padding_hw),
            "dilation_hw": list(self.dilation_hw),
            "groups": self.groups,
        }


@dataclass(frozen=True)
class SparseSpatialPlan:
    """Compact output lookup and deterministically propagated input support."""

    spec: Conv2DAdjointSpec
    output_indices: np.ndarray
    output_to_compact: np.ndarray
    input_indices: np.ndarray
    dense_fma_count: int
    sparse_fma_count: int

    def __post_init__(self) -> None:
        output_indices = np.ascontiguousarray(self.output_indices, dtype=np.int32).reshape(-1)
        output_to_compact = np.ascontiguousarray(
            self.output_to_compact, dtype=np.int32
        ).reshape(-1)
        input_indices = np.ascontiguousarray(self.input_indices, dtype=np.int32).reshape(-1)
        if output_to_compact.size != self.spec.output_hw[0] * self.spec.output_hw[1]:
            raise ValueError("output_to_compact has the wrong spatial size")
        if output_indices.size == 0 or input_indices.size == 0:
            raise ValueError("a sparse plan must retain at least one output and input site")
        if not np.array_equal(output_indices, np.unique(output_indices)):
            raise ValueError("output_indices must be unique and sorted")
        if not np.array_equal(input_indices, np.unique(input_indices)):
            raise ValueError("input_indices must be unique and sorted")
        if np.any(output_to_compact[output_indices] != np.arange(output_indices.size)):
            raise ValueError("output_to_compact is inconsistent with output_indices")
        if self.dense_fma_count <= 0 or not 0 < self.sparse_fma_count <= self.dense_fma_count:
            raise ValueError("invalid dense/sparse FMA accounting")
        output_indices.setflags(write=False)
        output_to_compact.setflags(write=False)
        input_indices.setflags(write=False)
        object.__setattr__(self, "output_indices", output_indices)
        object.__setattr__(self, "output_to_compact", output_to_compact)
        object.__setattr__(self, "input_indices", input_indices)

    @property
    def output_site_count(self) -> int:
        return int(self.output_indices.size)

    @property
    def input_site_count(self) -> int:
        return int(self.input_indices.size)

    @property
    def output_support_fraction(self) -> float:
        return self.output_site_count / float(self.output_to_compact.size)

    @property
    def input_support_fraction(self) -> float:
        return self.input_site_count / float(self.spec.input_hw[0] * self.spec.input_hw[1])

    @property
    def arithmetic_ceiling_x(self) -> float:
        return self.dense_fma_count / float(self.sparse_fma_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "output_site_count": self.output_site_count,
            "input_site_count": self.input_site_count,
            "output_support_fraction": self.output_support_fraction,
            "input_support_fraction": self.input_support_fraction,
            "dense_fma_count": self.dense_fma_count,
            "sparse_fma_count": self.sparse_fma_count,
            "dense_flops": 2 * self.dense_fma_count,
            "sparse_flops": 2 * self.sparse_fma_count,
            "arithmetic_ceiling_x": self.arithmetic_ceiling_x,
        }


def _valid_spatial_tap_count(mask: np.ndarray, spec: Conv2DAdjointSpec) -> int:
    count = 0
    active = np.argwhere(mask)
    for ho, wo in active:
        for kh in range(spec.kernel_hw[0]):
            hi = int(ho) * spec.stride_hw[0] + kh * spec.dilation_hw[0] - spec.padding_hw[0]
            if hi < 0 or hi >= spec.input_hw[0]:
                continue
            for kw in range(spec.kernel_hw[1]):
                wi = (
                    int(wo) * spec.stride_hw[1]
                    + kw * spec.dilation_hw[1]
                    - spec.padding_hw[1]
                )
                if 0 <= wi < spec.input_hw[1]:
                    count += 1
    return count


def build_sparse_spatial_plan(
    output_support: Any,
    spec: Conv2DAdjointSpec,
) -> SparseSpatialPlan:
    """Build a sorted compact lookup and exact convolution support propagation."""

    support = np.asarray(output_support, dtype=np.bool_)
    if support.shape != spec.output_hw:
        raise ValueError(f"output_support must have shape {spec.output_hw}, got {support.shape}")
    output_indices = np.flatnonzero(support.reshape(-1)).astype(np.int32)
    if output_indices.size == 0:
        raise ValueError("output_support must retain at least one site")
    output_to_compact = np.full(support.size, -1, dtype=np.int32)
    output_to_compact[output_indices] = np.arange(output_indices.size, dtype=np.int32)
    input_support = np.zeros(spec.input_hw, dtype=np.bool_)
    for output_index in output_indices:
        ho, wo = divmod(int(output_index), spec.output_hw[1])
        for kh in range(spec.kernel_hw[0]):
            hi = ho * spec.stride_hw[0] + kh * spec.dilation_hw[0] - spec.padding_hw[0]
            if hi < 0 or hi >= spec.input_hw[0]:
                continue
            for kw in range(spec.kernel_hw[1]):
                wi = wo * spec.stride_hw[1] + kw * spec.dilation_hw[1] - spec.padding_hw[1]
                if 0 <= wi < spec.input_hw[1]:
                    input_support[hi, wi] = True
    input_indices = np.flatnonzero(input_support.reshape(-1)).astype(np.int32)
    full_mask = np.ones(spec.output_hw, dtype=np.bool_)
    channel_fma = spec.cout * spec.input_channels_per_group
    dense_fma_count = _valid_spatial_tap_count(full_mask, spec) * channel_fma
    sparse_fma_count = _valid_spatial_tap_count(support, spec) * channel_fma
    return SparseSpatialPlan(
        spec=spec,
        output_indices=output_indices,
        output_to_compact=output_to_compact,
        input_indices=input_indices,
        dense_fma_count=dense_fma_count,
        sparse_fma_count=sparse_fma_count,
    )


def _float32(value: Any, *, name: str) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite float32 values")
    return result


def _validate_weight(weight: Any, spec: Conv2DAdjointSpec) -> np.ndarray:
    result = _float32(weight, name="weight")
    if result.shape != spec.weight_shape:
        raise ValueError(f"weight must have shape {spec.weight_shape}, got {result.shape}")
    return result


def _normalize_dense_cotangent(cotangent: Any, spec: Conv2DAdjointSpec) -> np.ndarray:
    cot = _float32(cotangent, name="cotangent")
    if cot.ndim == 3:
        cot = cot[None]
    expected = (*spec.output_hw, spec.cout)
    if cot.ndim != 4 or tuple(cot.shape[1:]) != expected:
        raise ValueError(f"cotangent must have shape (rank,{expected}), got {cot.shape}")
    if cot.shape[0] <= 0:
        raise ValueError("rank must be positive")
    return cot


def _normalize_compact_cotangent(cotangent: Any, plan: SparseSpatialPlan) -> np.ndarray:
    cot = _float32(cotangent, name="compact cotangent")
    if cot.ndim == 2:
        cot = cot[None]
    expected = (plan.output_site_count, plan.spec.cout)
    if cot.ndim != 3 or tuple(cot.shape[1:]) != expected:
        raise ValueError(f"compact cotangent must have shape (rank,{expected}), got {cot.shape}")
    if cot.shape[0] <= 0:
        raise ValueError("rank must be positive")
    return cot


def compact_cotangent(cotangent: Any, plan: SparseSpatialPlan) -> np.ndarray:
    """Gather a dense ``(rank,Hout,Wout,Cout)`` cotangent into plan order."""

    dense = _normalize_dense_cotangent(cotangent, plan.spec)
    flat = dense.reshape(dense.shape[0], -1, plan.spec.cout)
    return np.ascontiguousarray(flat[:, plan.output_indices, :], dtype=np.float32)


def _ordered_numpy_input_adjoint(
    compact: np.ndarray,
    weight: np.ndarray,
    plan: SparseSpatialPlan,
    target_indices: np.ndarray,
) -> np.ndarray:
    """Fixed-order NumPy core shared by the dense and compact authorities."""

    spec = plan.spec
    rank = int(compact.shape[0])
    result = np.zeros((rank, target_indices.size, spec.cin), dtype=np.float32)
    ipg = spec.input_channels_per_group
    opg = spec.output_channels_per_group
    for target_slot, input_index in enumerate(target_indices):
        hi, wi = divmod(int(input_index), spec.input_hw[1])
        for cin in range(spec.cin):
            group = cin // ipg
            local_in = cin - group * ipg
            acc = np.zeros(rank, dtype=np.float32)
            for kh in range(spec.kernel_hw[0]):
                numerator_h = hi + spec.padding_hw[0] - kh * spec.dilation_hw[0]
                if numerator_h < 0 or numerator_h % spec.stride_hw[0]:
                    continue
                ho = numerator_h // spec.stride_hw[0]
                if ho < 0 or ho >= spec.output_hw[0]:
                    continue
                for kw in range(spec.kernel_hw[1]):
                    numerator_w = wi + spec.padding_hw[1] - kw * spec.dilation_hw[1]
                    if numerator_w < 0 or numerator_w % spec.stride_hw[1]:
                        continue
                    wo = numerator_w // spec.stride_hw[1]
                    if wo < 0 or wo >= spec.output_hw[1]:
                        continue
                    compact_row = int(plan.output_to_compact[ho * spec.output_hw[1] + wo])
                    if compact_row < 0:
                        continue
                    for local_out in range(opg):
                        cout = group * opg + local_out
                        product = np.asarray(
                            compact[:, compact_row, cout] * weight[cout, kh, kw, local_in],
                            dtype=np.float32,
                        )
                        acc = np.asarray(acc + product, dtype=np.float32)
            result[:, target_slot, cin] = acc
    return result


def sparse_conv2d_input_adjoint_numpy_fp32(
    compact_cotangent_value: Any,
    weight: Any,
    plan: SparseSpatialPlan,
) -> np.ndarray:
    """Compact fixed-order NumPy-fp32 input adjoint authority."""

    compact = _normalize_compact_cotangent(compact_cotangent_value, plan)
    weight_array = _validate_weight(weight, plan.spec)
    return _ordered_numpy_input_adjoint(compact, weight_array, plan, plan.input_indices)


def dense_conv2d_input_adjoint_numpy_fp32(
    cotangent: Any,
    weight: Any,
    spec: Conv2DAdjointSpec,
) -> np.ndarray:
    """Dense fixed-order NumPy-fp32 authority in ``(rank,Hin,Win,Cin)`` layout."""

    dense = _normalize_dense_cotangent(cotangent, spec)
    weight_array = _validate_weight(weight, spec)
    full_plan = build_sparse_spatial_plan(np.ones(spec.output_hw, dtype=np.bool_), spec)
    compact = compact_cotangent(dense, full_plan)
    values = _ordered_numpy_input_adjoint(
        compact,
        weight_array,
        full_plan,
        np.arange(spec.input_hw[0] * spec.input_hw[1], dtype=np.int32),
    )
    return values.reshape(dense.shape[0], *spec.input_hw, spec.cin)


_SPARSE_INPUT_ADJOINT_SOURCE = r"""
    #pragma clang fp contract(off)
    uint elem = thread_position_in_grid.x;
    int rank = dims[0];
    int Hin  = dims[1];
    int Win  = dims[2];
    int Cin  = dims[3];
    int Hout = dims[4];
    int Wout = dims[5];
    int Cout = dims[6];
    int Sout = dims[7];
    int Sin  = dims[8];
    if (elem >= (uint)(Sin * Cin)) return;

    int input_slot = elem / Cin;
    int cin = elem - input_slot * Cin;
    int input_linear = input_idx[input_slot];
    int hi = input_linear / Win;
    int wi = input_linear - hi * Win;

    int kH = conv[0];
    int kW = conv[1];
    int Ipg = conv[2];
    int groups = conv[3];
    int stride_h = conv[4];
    int stride_w = conv[5];
    int pad_h = conv[6];
    int pad_w = conv[7];
    int dil_h = conv[8];
    int dil_w = conv[9];
    int Opg = Cout / groups;
    int group = cin / Ipg;
    int local_in = cin - group * Ipg;

    float acc[8];
    for (int r = 0; r < rank; ++r) acc[r] = 0.0f;

    for (int kh = 0; kh < kH; ++kh) {
        int numerator_h = hi + pad_h - kh * dil_h;
        if (numerator_h < 0 || numerator_h % stride_h != 0) continue;
        int ho = numerator_h / stride_h;
        if (ho < 0 || ho >= Hout) continue;
        for (int kw = 0; kw < kW; ++kw) {
            int numerator_w = wi + pad_w - kw * dil_w;
            if (numerator_w < 0 || numerator_w % stride_w != 0) continue;
            int wo = numerator_w / stride_w;
            if (wo < 0 || wo >= Wout) continue;
            int compact_row = out_map[ho * Wout + wo];
            if (compact_row < 0) continue;
            for (int local_out = 0; local_out < Opg; ++local_out) {
                int cout = group * Opg + local_out;
                int weight_index = ((cout * kH + kh) * kW + kw) * Ipg + local_in;
                float weight_value = wt[weight_index];
                for (int r = 0; r < rank; ++r) {
                    int cot_index = (r * Sout + compact_row) * Cout + cout;
                    float product = weight_value * cot[cot_index];
                    acc[r] = acc[r] + product;
                }
            }
        }
    }
    for (int r = 0; r < rank; ++r) {
        int output_index = (r * Sin + input_slot) * Cin + cin;
        gx[output_index] = acc[r];
    }
"""

_METAL_KERNEL: Any | None = None


def _metal_kernel() -> Any:
    global _METAL_KERNEL
    if _METAL_KERNEL is None:
        import mlx.core as mx

        _METAL_KERNEL = mx.fast.metal_kernel(
            name=SPARSE_ADJOINT_KERNEL_NAME,
            input_names=["cot", "wt", "out_map", "input_idx", "dims", "conv"],
            output_names=["gx"],
            source=_SPARSE_INPUT_ADJOINT_SOURCE,
        )
    return _METAL_KERNEL


def sparse_adjoint_metal_available() -> bool:
    """Return true only when a tiny MLX array can be evaluated on Metal."""

    try:
        import mlx.core as mx

        if mx.default_device().type != mx.gpu:
            return False
        probe = mx.array([0.0], dtype=mx.float32)
        mx.eval(probe)
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class MetalSparseAdjointExecutable:
    """Prepared MLX arrays for repeated, synchronization-controlled timing."""

    compact_cotangent: Any
    weight: Any
    output_to_compact: Any
    input_indices: Any
    dims: Any
    conv: Any
    output_shape: tuple[int, int, int]

    def __call__(self) -> Any:
        (result,) = _metal_kernel()(
            inputs=[
                self.compact_cotangent,
                self.weight,
                self.output_to_compact,
                self.input_indices,
                self.dims,
                self.conv,
            ],
            output_shapes=[self.output_shape],
            output_dtypes=[self.compact_cotangent.dtype],
            grid=(self.output_shape[1] * self.output_shape[2], 1, 1),
            threadgroup=(256, 1, 1),
        )
        return result


def make_sparse_conv2d_input_adjoint_metal(
    compact_cotangent_value: Any,
    weight: Any,
    plan: SparseSpatialPlan,
) -> MetalSparseAdjointExecutable:
    """Prepare one fused-rank (``rank <= 8``) compact Metal launch."""

    import mlx.core as mx

    compact = _normalize_compact_cotangent(compact_cotangent_value, plan)
    if compact.shape[0] > MAX_FUSED_RANK:
        raise ValueError(
            f"prepared launch rank {compact.shape[0]} exceeds MAX_FUSED_RANK={MAX_FUSED_RANK}; "
            "use sparse_conv2d_input_adjoint_metal for deterministic chunking"
        )
    weight_array = _validate_weight(weight, plan.spec)
    spec = plan.spec
    dims = np.array(
        [
            compact.shape[0],
            spec.input_hw[0],
            spec.input_hw[1],
            spec.cin,
            spec.output_hw[0],
            spec.output_hw[1],
            spec.cout,
            plan.output_site_count,
            plan.input_site_count,
        ],
        dtype=np.int32,
    )
    conv = np.array(
        [
            spec.kernel_hw[0],
            spec.kernel_hw[1],
            spec.input_channels_per_group,
            spec.groups,
            spec.stride_hw[0],
            spec.stride_hw[1],
            spec.padding_hw[0],
            spec.padding_hw[1],
            spec.dilation_hw[0],
            spec.dilation_hw[1],
        ],
        dtype=np.int32,
    )
    return MetalSparseAdjointExecutable(
        compact_cotangent=mx.array(compact, dtype=mx.float32),
        weight=mx.array(weight_array, dtype=mx.float32),
        output_to_compact=mx.array(plan.output_to_compact, dtype=mx.int32),
        input_indices=mx.array(plan.input_indices, dtype=mx.int32),
        dims=mx.array(dims, dtype=mx.int32),
        conv=mx.array(conv, dtype=mx.int32),
        output_shape=(int(compact.shape[0]), plan.input_site_count, spec.cin),
    )


def sparse_conv2d_input_adjoint_metal(
    compact_cotangent_value: Any,
    weight: Any,
    plan: SparseSpatialPlan,
) -> Any:
    """Run the compact Metal adjoint, chunking rank deterministically above eight."""

    import mlx.core as mx

    compact = _normalize_compact_cotangent(compact_cotangent_value, plan)
    outputs = []
    for start in range(0, compact.shape[0], MAX_FUSED_RANK):
        executable = make_sparse_conv2d_input_adjoint_metal(
            compact[start : start + MAX_FUSED_RANK], weight, plan
        )
        outputs.append(executable())
    return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=0)
