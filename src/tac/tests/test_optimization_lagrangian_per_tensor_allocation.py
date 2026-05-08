"""Unit tests for ``tac.optimization.lagrangian_per_tensor_allocation``."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tac.codec.cost_curves import (
    TensorBlob,
    precompute_per_tensor_K_curves,
    precompute_per_tensor_sparsity_curves,
)
from tac.optimization.lagrangian_per_tensor_allocation import (
    AllocationResult,
    JacobianWeightedAllocator,
    LagrangianPerTensorAllocator,
    UniwardWeightedAllocator,
    compute_jacobian_importance_weights,
    compute_local_variance_proxy,
    compute_uniward_weights,
    normalize_importance_weights,
)

# ---------------------------------------------------------------------------
# Allocate at fixed λ
# ---------------------------------------------------------------------------


def test_allocate_large_lambda_picks_lossless() -> None:
    curves = [
        [
            {"alpha": 0.0, "bytes": 100, "rel_err": 0.0},
            {"alpha": 0.5, "bytes": 60, "rel_err": 0.5},
        ],
        [
            {"alpha": 0.0, "bytes": 80, "rel_err": 0.0},
            {"alpha": 0.5, "bytes": 40, "rel_err": 0.5},
        ],
    ]
    alloc = LagrangianPerTensorAllocator()
    res = alloc.allocate(curves, lam=1e9)
    # Large λ pushes everyone to rel_err=0
    assert all(s["alpha"] == 0.0 for s in res.selections)
    assert res.total_bytes == 180
    assert res.rel_err == 0.0


def test_allocate_zero_lambda_picks_smallest_bytes() -> None:
    curves = [
        [
            {"alpha": 0.0, "bytes": 100, "rel_err": 0.0},
            {"alpha": 0.5, "bytes": 60, "rel_err": 0.5},
        ],
    ]
    alloc = LagrangianPerTensorAllocator()
    res = alloc.allocate(curves, lam=0.0)
    assert res.selections[0]["alpha"] == 0.5
    assert res.total_bytes == 60


def test_allocate_returns_allocation_result_with_to_dict() -> None:
    curves = [[{"alpha": 0.0, "bytes": 50, "rel_err": 0.0}]]
    res = LagrangianPerTensorAllocator().allocate(curves, lam=1.0)
    assert isinstance(res, AllocationResult)
    d = res.to_dict()
    assert d["total_bytes"] == 50
    assert d["lambda"] == 1.0
    assert "selections" in d


# ---------------------------------------------------------------------------
# λ bisection
# ---------------------------------------------------------------------------


def _build_real_curves(n_tensors: int = 3, seed: int = 0) -> list[list[dict]]:
    rng = np.random.default_rng(seed)
    blobs = [
        TensorBlob(name=f"t{i}", raw=rng.integers(-127, 128, size=128, dtype=np.int32))
        for i in range(n_tensors)
    ]
    return precompute_per_tensor_sparsity_curves(blobs, alphas=[0.3, 0.5, 0.7, 0.9])


def test_bisect_meets_target_or_lossless() -> None:
    curves = _build_real_curves()
    res = LagrangianPerTensorAllocator().bisect_for_rms_target(curves, rms_target=0.1)
    assert res.rel_err <= 0.1 + 1e-9


def test_bisect_with_zero_target_returns_lossless() -> None:
    curves = _build_real_curves()
    res = LagrangianPerTensorAllocator().bisect_for_rms_target(curves, rms_target=0.0)
    assert res.rel_err == 0.0
    expected = sum(tc[0]["bytes"] for tc in curves)
    assert res.total_bytes == expected


# ---------------------------------------------------------------------------
# Joint encoder hook
# ---------------------------------------------------------------------------


def test_allocate_with_joint_encoder_uses_hook_bytes() -> None:
    curves = [
        [
            {"alpha": 0.0, "bytes": 50, "rel_err": 0.0},
            {"alpha": 0.5, "bytes": 30, "rel_err": 0.2},
        ],
    ]

    def joint_hook(selections: list[dict]) -> dict[str, Any]:
        # Pretend joint encoding is half of summed per-tensor bytes
        return {
            "total_bytes": sum(s["bytes"] for s in selections) // 2,
            "rel_err": float(np.sqrt(np.mean([s["rel_err"] ** 2 for s in selections]))),
            "joint_signature": "test",
        }

    alloc = LagrangianPerTensorAllocator(joint_encoder=joint_hook)
    res = alloc.allocate(curves, lam=0.0)
    assert res.total_bytes == 15  # 30 // 2
    assert res.joint_extras["joint_signature"] == "test"
    d = res.to_dict()
    assert d["joint_signature"] == "test"


# ---------------------------------------------------------------------------
# K curves
# ---------------------------------------------------------------------------


def test_allocator_works_on_K_curves() -> None:
    rng = np.random.default_rng(7)
    blobs = [
        TensorBlob(name=f"t{i}", raw=rng.integers(-127, 128, size=128, dtype=np.int32))
        for i in range(3)
    ]
    K_curves = precompute_per_tensor_K_curves(blobs)
    alloc = LagrangianPerTensorAllocator()
    res = alloc.allocate(K_curves, lam=1e9)
    # Large λ → K=1 for everyone
    assert all(s["K"] == 1 for s in res.selections)


# ---------------------------------------------------------------------------
# UNIWARD weights
# ---------------------------------------------------------------------------


def test_compute_local_variance_proxy() -> None:
    syms = [
        np.array([0, 0, 0, 0], dtype=np.int32),
        np.array([10, -10, 10, -10], dtype=np.int32),
    ]
    var = compute_local_variance_proxy(syms)
    assert var[0] == 0.0
    assert var[1] == pytest.approx(100.0, rel=1e-6)


def test_compute_uniward_weights_inverse_variance_ordering() -> None:
    weights = compute_uniward_weights([0.0, 100.0])
    assert weights[0] > weights[1]


def test_uniward_allocator_allocates_more_error_to_high_variance() -> None:
    rng = np.random.default_rng(13)
    high_var_syms = rng.integers(-127, 128, size=128, dtype=np.int32)
    low_var_syms = rng.integers(-2, 3, size=128, dtype=np.int32)  # near zero
    blobs = [
        TensorBlob(name="hi", raw=high_var_syms),
        TensorBlob(name="lo", raw=low_var_syms),
    ]
    K_curves = precompute_per_tensor_K_curves(blobs)
    alloc = UniwardWeightedAllocator([b.raw for b in blobs])
    res = alloc.bisect_for_rms_target(K_curves, rms_target=0.05)
    # Inverse-variance weights ⇒ low-variance tensor gets a high λ·w·rel_err²
    # penalty, biasing it toward smaller K (less error). Verify the
    # high-variance tensor's chosen K is ≥ low-variance tensor's K (i.e. it
    # absorbs at least as much error).
    high_K = res.selections[0]["K"]
    low_K = res.selections[1]["K"]
    assert high_K >= low_K


def test_uniward_allocator_exposes_variances() -> None:
    syms = [np.array([10, -10], dtype=np.int32), np.array([1, -1], dtype=np.int32)]
    alloc = UniwardWeightedAllocator(syms)
    assert len(alloc.variances) == 2
    assert alloc.variances[0] > alloc.variances[1]


# ---------------------------------------------------------------------------
# Jacobian-pullback weights
# ---------------------------------------------------------------------------


def test_normalize_importance_weights_mean_one() -> None:
    weights = normalize_importance_weights([10.0, 1.0, 4.0])
    assert np.mean(weights) == pytest.approx(1.0, rel=1e-12)
    assert weights[0] > weights[2] > weights[1]


def test_compute_jacobian_importance_weights_texture_capacity() -> None:
    weights = compute_jacobian_importance_weights(
        [1.0, 1.0],
        texture_capacity=[100.0, 1.0],
    )
    assert np.mean(weights) == pytest.approx(1.0, rel=1e-12)
    # Higher texture capacity lowers the protection penalty.
    assert weights[0] < weights[1]


def test_compute_jacobian_importance_rejects_all_zero() -> None:
    with pytest.raises(ValueError, match="at least one positive"):
        compute_jacobian_importance_weights([0.0, 0.0])


def test_normalize_importance_rejects_nonfinite_floor() -> None:
    with pytest.raises(ValueError, match="floor must be finite"):
        normalize_importance_weights([1.0, 2.0], floor=float("nan"))


def test_jacobian_allocator_protects_high_importance_tensor() -> None:
    curves = [
        [
            {"K": 1, "byte_proxy": 100, "rel_err": 0.0},
            {"K": 8, "byte_proxy": 50, "rel_err": 0.5},
        ],
        [
            {"K": 1, "byte_proxy": 100, "rel_err": 0.0},
            {"K": 8, "byte_proxy": 50, "rel_err": 0.5},
        ],
    ]
    alloc = JacobianWeightedAllocator([10.0, 1.0])
    res = alloc.allocate(curves, lam=160.0)
    assert np.mean(alloc.importance_weights) == pytest.approx(1.0, rel=1e-12)
    assert res.selections[0]["K"] == 1
    assert res.selections[1]["K"] == 8


# ---------------------------------------------------------------------------
# Memoization correctness
# ---------------------------------------------------------------------------


def test_bisect_memoize_does_not_change_result() -> None:
    curves = _build_real_curves(seed=21)
    a = LagrangianPerTensorAllocator()
    res_memo = a.bisect_for_rms_target(curves, rms_target=0.05, memoize=True)
    res_nomemo = a.bisect_for_rms_target(curves, rms_target=0.05, memoize=False)
    assert res_memo.total_bytes == res_nomemo.total_bytes
    assert res_memo.rel_err == pytest.approx(res_nomemo.rel_err, rel=1e-6, abs=1e-9)


# ---------------------------------------------------------------------------
# Weight length validation
# ---------------------------------------------------------------------------


def test_weights_length_mismatch_raises() -> None:
    curves = _build_real_curves(n_tensors=3)
    alloc = LagrangianPerTensorAllocator(weights=[1.0, 1.0])  # only 2
    with pytest.raises(ValueError):
        alloc.allocate(curves, lam=1.0)
