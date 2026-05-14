# SPDX-License-Identifier: MIT
"""Pinning tests for canonical ``rel_err`` definition + allocator-side guard.

The canonical definition is RMS = ``sqrt(mean(diff²)) / sqrt(mean(orig²+ε))``.
L1 (sum/sum) and max forms are available as explicit opt-ins. The Lagrangian
allocator asserts form-uniformity at the entry of its bisection.

See ``.omx/research/rel_err_inconsistency_audit_20260508_claude.md`` and
``feedback_rel_err_l1_rms_canonicalization_20260508.md``.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from tac.codec.cost_curves import (
    precompute_per_tensor_K_curves,
    precompute_per_tensor_sparsity_curves,
)
from tac.codec.per_tensor_codecs import (
    REL_ERR_FORM_LOSSY_K,
    REL_ERR_FORM_SPARSITY,
)
from tac.codec.rel_err import (
    EPS_NUMERICAL,
    REL_ERR_FORM_KEY,
    RelErrForm,
    aggregate_rel_err,
    assert_uniform_rel_err_form,
    compute_rel_err,
)


# ---------------------------------------------------------------------------
# Pin the canonical RMS definition with a known reference
# ---------------------------------------------------------------------------


class TestComputeRelErrRMS:
    def test_zero_when_identical(self):
        x = np.array([1.0, 2.0, 3.0])
        assert compute_rel_err(x, x) == 0.0

    def test_zero_when_orig_is_zero(self):
        # eps-floor produces a finite (zero) rel_err when orig is exactly 0
        x = np.zeros(5)
        y = np.zeros(5)
        assert compute_rel_err(x, y) == 0.0

    def test_known_value_unit_vector(self):
        # diff = [0.1, 0, 0, 0]; orig = [1, 0, 0, 0]
        # RMS-numerator = sqrt(mean([0.01, 0, 0, 0])) = sqrt(0.0025) = 0.05
        # RMS-denominator = sqrt(mean([1, 0, 0, 0])) = sqrt(0.25) = 0.5
        # ratio = 0.1
        orig = np.array([1.0, 0.0, 0.0, 0.0])
        recon = np.array([0.9, 0.0, 0.0, 0.0])
        got = compute_rel_err(recon, orig, mode=RelErrForm.RMS)
        assert math.isclose(got, 0.1, abs_tol=1e-12)

    def test_default_mode_is_rms(self):
        orig = np.array([1.0, 0.0, 0.0, 0.0])
        recon = np.array([0.9, 0.0, 0.0, 0.0])
        assert compute_rel_err(recon, orig) == compute_rel_err(
            recon, orig, mode=RelErrForm.RMS
        )

    def test_string_mode_works(self):
        orig = np.array([1.0, 0.0])
        recon = np.array([0.9, 0.0])
        a = compute_rel_err(recon, orig, mode="rms")
        b = compute_rel_err(recon, orig, mode=RelErrForm.RMS)
        assert a == b

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown rel_err mode"):
            compute_rel_err(np.zeros(2), np.zeros(2), mode="hypotenuse")

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            compute_rel_err(np.zeros(2), np.zeros(3))


# ---------------------------------------------------------------------------
# Smoke-test L1 / L2 / max alternatives
# ---------------------------------------------------------------------------


class TestComputeRelErrAlternatives:
    def test_l1_ratio_simple(self):
        orig = np.array([1.0, 2.0, 3.0])
        recon = np.array([1.0, 2.0, 3.5])
        # diff abs sum = 0.5; orig abs sum = 6.0
        got = compute_rel_err(recon, orig, mode=RelErrForm.L1_RATIO)
        assert math.isclose(got, 0.5 / 6.0, abs_tol=1e-12)

    def test_l2_ratio_matches_norm(self):
        orig = np.array([3.0, 4.0])
        recon = np.array([3.0, 4.5])
        # ‖diff‖₂ = 0.5; ‖orig‖₂ = 5
        got = compute_rel_err(recon, orig, mode=RelErrForm.L2_RATIO)
        assert math.isclose(got, 0.5 / (5.0 + EPS_NUMERICAL), abs_tol=1e-12)

    def test_max_ratio_picks_worst_element(self):
        orig = np.array([10.0, 0.5])
        recon = np.array([10.0, 1.5])
        # max|diff| = 1.0; max|orig| = 10
        got = compute_rel_err(recon, orig, mode=RelErrForm.MAX_RATIO)
        assert math.isclose(got, 0.1, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Aggregator: RMS² semantics consistent with allocator's squared penalty
# ---------------------------------------------------------------------------


class TestAggregateRelErr:
    def test_rms_aggregate_matches_sqrt_mean_squared(self):
        per_tensor = [0.1, 0.2, 0.3]
        got = aggregate_rel_err(per_tensor, mode=RelErrForm.RMS)
        expect = math.sqrt(np.mean(np.array(per_tensor) ** 2))
        assert math.isclose(got, expect, abs_tol=1e-12)

    def test_l1_aggregate_is_mean_not_sum(self):
        per_tensor = [0.1, 0.2, 0.3]
        # mean (not sum) — keeps output in [0, ~1] ratio range
        got = aggregate_rel_err(per_tensor, mode=RelErrForm.L1_RATIO)
        assert math.isclose(got, 0.2, abs_tol=1e-12)

    def test_max_aggregate(self):
        per_tensor = [0.1, 0.7, 0.3]
        got = aggregate_rel_err(per_tensor, mode=RelErrForm.MAX_RATIO)
        assert got == 0.7

    def test_empty_returns_zero(self):
        assert aggregate_rel_err([]) == 0.0

    def test_negative_value_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            aggregate_rel_err([0.1, -0.05])

    def test_lagrangian_squared_penalty_consistency(self):
        # The allocator computes cost = bytes + λ · rel_err².
        # For RMS aggregate, sum of per-tensor rel_err² used as the
        # bisection target equals mean(rel_err_t²) · N = N · RMS²,
        # so the squared penalty IS proportional to N · RMS².
        per_tensor = [0.05, 0.1, 0.15]
        rms = aggregate_rel_err(per_tensor, mode=RelErrForm.RMS)
        n = len(per_tensor)
        sum_sq = sum(e ** 2 for e in per_tensor)
        # The relation: sum_sq = n * RMS²
        assert math.isclose(sum_sq, n * rms ** 2, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Curve emission tags carry the form
# ---------------------------------------------------------------------------


class TestCurveFormTags:
    def test_sparsity_curves_carry_l2_tag(self):
        rng = np.random.default_rng(42)
        symbols = rng.integers(-127, 127, size=200, dtype=np.int8)
        curves = precompute_per_tensor_sparsity_curves(
            [("t0", symbols)], alphas=[0.3, 0.5]
        )
        for row in curves[0]:
            assert REL_ERR_FORM_KEY in row
            assert row[REL_ERR_FORM_KEY] == REL_ERR_FORM_SPARSITY

    def test_K_curves_carry_l1_tag(self):
        rng = np.random.default_rng(43)
        symbols = rng.integers(-127, 127, size=200, dtype=np.int8)
        from tac.codec.cost_curves import TensorBlob

        tensors = [TensorBlob(name="t0", raw=symbols)]
        curves = precompute_per_tensor_K_curves(tensors, K_range=[1, 2, 4])
        for row in curves[0]:
            assert REL_ERR_FORM_KEY in row
            assert row[REL_ERR_FORM_KEY] == REL_ERR_FORM_LOSSY_K


# ---------------------------------------------------------------------------
# Allocator form-uniformity guard
# ---------------------------------------------------------------------------


class TestAllocatorFormUniformity:
    def test_uniform_form_passes(self):
        from tac.codec.cost_curves import TensorBlob, precompute_per_tensor_K_curves
        from tac.optimization.lagrangian_per_tensor_allocation import (
            LagrangianPerTensorAllocator,
        )

        rng = np.random.default_rng(44)
        tensors = [
            TensorBlob(name="t0", raw=rng.integers(-127, 127, size=200, dtype=np.int8)),
            TensorBlob(name="t1", raw=rng.integers(-127, 127, size=200, dtype=np.int8)),
        ]
        curves = precompute_per_tensor_K_curves(tensors, K_range=[1, 2, 4])
        allocator = LagrangianPerTensorAllocator()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = allocator.bisect_for_rms_target(curves, rms_target=0.2)
        assert res.total_bytes > 0
        assert res.joint_extras.get(REL_ERR_FORM_KEY) == REL_ERR_FORM_LOSSY_K

    def test_mixed_form_raises(self):
        from tac.optimization.lagrangian_per_tensor_allocation import (
            LagrangianPerTensorAllocator,
        )

        # Build curves with mixed form tags — should raise.
        curves = [
            [
                {"K": 1, "byte_proxy": 100, "rel_err": 0.0, REL_ERR_FORM_KEY: "l1_ratio"},
                {"K": 2, "byte_proxy": 90, "rel_err": 0.1, REL_ERR_FORM_KEY: "l1_ratio"},
            ],
            [
                {"K": 1, "byte_proxy": 100, "rel_err": 0.0, REL_ERR_FORM_KEY: "rms"},
                {"K": 2, "byte_proxy": 90, "rel_err": 0.05, REL_ERR_FORM_KEY: "rms"},
            ],
        ]
        allocator = LagrangianPerTensorAllocator()
        with pytest.raises(ValueError, match="mixed rel_err_form"):
            allocator.bisect_for_rms_target(curves, rms_target=0.2)

    def test_untagged_curves_warn_lax_mode(self):
        from tac.optimization.lagrangian_per_tensor_allocation import (
            LagrangianPerTensorAllocator,
        )

        # Untagged curves (legacy callers) — warn once, do not raise.
        curves = [
            [
                {"K": 1, "byte_proxy": 100, "rel_err": 0.0},
                {"K": 2, "byte_proxy": 90, "rel_err": 0.05},
            ],
        ]
        allocator = LagrangianPerTensorAllocator()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            allocator.bisect_for_rms_target(curves, rms_target=0.2)
        assert any(
            "rel_err_form" in str(w.message) or "rel_err" in str(w.message)
            for w in caught
        ), [str(w.message) for w in caught]

    def test_untagged_curves_strict_raises(self):
        from tac.optimization.lagrangian_per_tensor_allocation import (
            LagrangianPerTensorAllocator,
        )

        curves = [
            [
                {"K": 1, "byte_proxy": 100, "rel_err": 0.0},
                {"K": 2, "byte_proxy": 90, "rel_err": 0.05},
            ],
        ]
        allocator = LagrangianPerTensorAllocator(strict_rel_err_form=True)
        with pytest.raises(ValueError):
            allocator.bisect_for_rms_target(curves, rms_target=0.2)


# ---------------------------------------------------------------------------
# Bare assert_uniform_rel_err_form contract
# ---------------------------------------------------------------------------


class TestAssertUniformRelErrForm:
    def test_empty_returns_none(self):
        assert assert_uniform_rel_err_form([]) is None

    def test_uniform_returns_form(self):
        curves = [
            [{"rel_err": 0.0, REL_ERR_FORM_KEY: "rms"}],
            [{"rel_err": 0.1, REL_ERR_FORM_KEY: "rms"}],
        ]
        got = assert_uniform_rel_err_form(curves)
        assert got is RelErrForm.RMS

    def test_mixed_raises_in_lax_and_strict(self):
        curves = [
            [{"rel_err": 0.0, REL_ERR_FORM_KEY: "rms"}],
            [{"rel_err": 0.1, REL_ERR_FORM_KEY: "l1_ratio"}],
        ]
        with pytest.raises(ValueError, match="mixed"):
            assert_uniform_rel_err_form(curves, strict=False)
        with pytest.raises(ValueError, match="mixed"):
            assert_uniform_rel_err_form(curves, strict=True)

    def test_partial_missing_lax_returns_seen_form(self):
        curves = [
            [{"rel_err": 0.0, REL_ERR_FORM_KEY: "rms"}],
            [{"rel_err": 0.1}],  # missing tag
        ]
        got = assert_uniform_rel_err_form(curves, strict=False)
        assert got is RelErrForm.RMS

    def test_partial_missing_strict_raises(self):
        curves = [
            [{"rel_err": 0.0, REL_ERR_FORM_KEY: "rms"}],
            [{"rel_err": 0.1}],
        ]
        with pytest.raises(ValueError, match="strict=True"):
            assert_uniform_rel_err_form(curves, strict=True)
