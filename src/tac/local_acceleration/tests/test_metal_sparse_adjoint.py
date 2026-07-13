# SPDX-License-Identifier: MIT
"""Parity and support-accounting gates for the compact sparse adjoint."""

from __future__ import annotations

import numpy as np
import pytest

from tac.local_acceleration.metal_sparse_adjoint import (
    _SPARSE_INPUT_ADJOINT_SOURCE,
    MAX_FUSED_RANK,
    Conv2DAdjointSpec,
    build_sparse_spatial_plan,
    compact_cotangent,
    dense_conv2d_input_adjoint_numpy_fp32,
    sparse_adjoint_metal_available,
    sparse_conv2d_input_adjoint_metal,
    sparse_conv2d_input_adjoint_numpy_fp32,
)

CASES = (
    Conv2DAdjointSpec(
        input_hw=(7, 8),
        output_hw=(7, 8),
        cin=3,
        cout=5,
        kernel_hw=(3, 3),
        padding_hw=(1, 1),
    ),
    Conv2DAdjointSpec(
        input_hw=(8, 9),
        output_hw=(4, 5),
        cin=4,
        cout=6,
        kernel_hw=(3, 3),
        stride_hw=(2, 2),
        padding_hw=(1, 1),
        groups=2,
    ),
    Conv2DAdjointSpec(
        input_hw=(9, 7),
        output_hw=(5, 4),
        cin=5,
        cout=5,
        kernel_hw=(5, 5),
        stride_hw=(2, 2),
        padding_hw=(2, 2),
        groups=5,
    ),
)


def _fixture(
    spec: Conv2DAdjointSpec, *, rank: int = 2, seed: int = 486
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mask = np.zeros(spec.output_hw, dtype=np.bool_)
    flat = mask.reshape(-1)
    flat[(np.arange(max(1, flat.size // 4)) * flat.size) // max(1, flat.size // 4)] = True
    cotangent = rng.standard_normal((rank, *spec.output_hw, spec.cout)).astype(np.float32)
    weight = rng.standard_normal(spec.weight_shape).astype(np.float32)
    return mask, cotangent, weight


@pytest.mark.parametrize("spec", CASES)
def test_numpy_compact_adjoint_is_bit_identical_to_dense_on_support(
    spec: Conv2DAdjointSpec,
) -> None:
    mask, cotangent, weight = _fixture(spec)
    plan = build_sparse_spatial_plan(mask, spec)
    compact = compact_cotangent(cotangent, plan)
    sparse = sparse_conv2d_input_adjoint_numpy_fp32(compact, weight, plan)
    dense_masked = dense_conv2d_input_adjoint_numpy_fp32(
        cotangent * mask[None, :, :, None], weight, spec
    )
    restricted = dense_masked.reshape(cotangent.shape[0], -1, spec.cin)[
        :, plan.input_indices, :
    ]
    assert np.array_equal(sparse, restricted)


@pytest.mark.parametrize("spec", CASES)
def test_numpy_dense_authority_matches_independent_torch_autograd(
    spec: Conv2DAdjointSpec,
) -> None:
    import torch
    import torch.nn.functional as torch_functional

    _, cotangent, weight = _fixture(spec, rank=2)
    x = torch.zeros(
        (cotangent.shape[0], spec.cin, *spec.input_hw),
        dtype=torch.float32,
        requires_grad=True,
    )
    torch_weight = torch.from_numpy(weight.transpose(0, 3, 1, 2).copy())
    y = torch_functional.conv2d(
        x,
        torch_weight,
        stride=spec.stride_hw,
        padding=spec.padding_hw,
        dilation=spec.dilation_hw,
        groups=spec.groups,
    )
    torch_cotangent = torch.from_numpy(cotangent.transpose(0, 3, 1, 2).copy())
    (torch_adjoint,) = torch.autograd.grad(y, x, grad_outputs=torch_cotangent)
    expected = dense_conv2d_input_adjoint_numpy_fp32(cotangent, weight, spec)
    observed = torch_adjoint.detach().numpy().transpose(0, 2, 3, 1)
    np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-6)


def test_rank_above_fused_width_preserves_numpy_chunk_identity() -> None:
    spec = CASES[0]
    mask, cotangent, weight = _fixture(spec, rank=MAX_FUSED_RANK + 3)
    plan = build_sparse_spatial_plan(mask, spec)
    compact = compact_cotangent(cotangent, plan)
    whole = sparse_conv2d_input_adjoint_numpy_fp32(compact, weight, plan)
    chunked = np.concatenate(
        [
            sparse_conv2d_input_adjoint_numpy_fp32(
                compact[start : start + MAX_FUSED_RANK], weight, plan
            )
            for start in range(0, compact.shape[0], MAX_FUSED_RANK)
        ],
        axis=0,
    )
    assert np.array_equal(whole, chunked)


def test_support_propagation_and_fma_accounting_are_exact() -> None:
    spec = Conv2DAdjointSpec(
        input_hw=(5, 5),
        output_hw=(5, 5),
        cin=2,
        cout=3,
        kernel_hw=(3, 3),
        padding_hw=(1, 1),
    )
    mask = np.zeros(spec.output_hw, dtype=np.bool_)
    mask[2, 2] = True
    plan = build_sparse_spatial_plan(mask, spec)
    expected = np.array(
        [
            1 * 5 + 1,
            1 * 5 + 2,
            1 * 5 + 3,
            2 * 5 + 1,
            2 * 5 + 2,
            2 * 5 + 3,
            3 * 5 + 1,
            3 * 5 + 2,
            3 * 5 + 3,
        ],
        dtype=np.int32,
    )
    assert np.array_equal(plan.input_indices, expected)
    assert plan.sparse_fma_count == 3 * 3 * 3 * 2
    assert plan.dense_fma_count == (9 * 9 + 12 * 6 + 4 * 4) * 3 * 2
    assert plan.arithmetic_ceiling_x == pytest.approx(
        plan.dense_fma_count / plan.sparse_fma_count
    )


def test_invalid_geometry_and_empty_support_fail_closed() -> None:
    with pytest.raises(ValueError, match="disagrees"):
        Conv2DAdjointSpec(
            input_hw=(5, 5),
            output_hw=(2, 2),
            cin=1,
            cout=1,
            kernel_hw=(3, 3),
            padding_hw=(1, 1),
        )
    spec = CASES[0]
    with pytest.raises(ValueError, match="at least one"):
        build_sparse_spatial_plan(np.zeros(spec.output_hw, dtype=np.bool_), spec)


def test_metal_source_preserves_fixed_order_no_atomic_contract() -> None:
    assert "#pragma clang fp contract(off)" in _SPARSE_INPUT_ADJOINT_SOURCE
    assert "atomic" not in _SPARSE_INPUT_ADJOINT_SOURCE.lower()
    assert _SPARSE_INPUT_ADJOINT_SOURCE.index("for (int kh") < (
        _SPARSE_INPUT_ADJOINT_SOURCE.index("for (int kw")
    )
    assert _SPARSE_INPUT_ADJOINT_SOURCE.index("for (int kw") < (
        _SPARSE_INPUT_ADJOINT_SOURCE.index("for (int local_out")
    )


@pytest.mark.skipif(not sparse_adjoint_metal_available(), reason="Metal device unavailable")
@pytest.mark.parametrize("spec", CASES)
def test_metal_is_bit_identical_to_numpy_dense_on_support(spec: Conv2DAdjointSpec) -> None:
    import mlx.core as mx

    mask, cotangent, weight = _fixture(spec)
    plan = build_sparse_spatial_plan(mask, spec)
    compact = compact_cotangent(cotangent, plan)
    expected = sparse_conv2d_input_adjoint_numpy_fp32(compact, weight, plan)
    observed_mx = sparse_conv2d_input_adjoint_metal(compact, weight, plan)
    mx.eval(observed_mx)
    observed = np.asarray(observed_mx)
    assert np.array_equal(observed, expected)
