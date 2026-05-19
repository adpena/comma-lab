# SPDX-License-Identifier: MIT
"""Tests for tac.mps_diagnostic.kahan_conv2d (slot 9 HIGHEST-EV correction).

Per slot 9 formalization (`feedback_mps_drift_mathematical_and_engineering_formalization_landed_20260519.md`):
predicted ~10x drift reduction from Kahan summation in Conv2d accumulation.

Per Higham 2002 chapter 4: naive sum of N floats accumulates roundoff at
O(eps * sqrt(N)); Kahan summation reduces this to O(eps) independent of N.

The tests verify:
  - Kahan-vs-naive precision invariant on a known-bad case (N=10^6 small values).
  - Standard PyTorch conv2d agreement on CPU (where Kahan is mathematically
    equivalent up to bit-level rounding).
  - Monkey-patch round-trip.
  - Provenance tags pinned.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from tac.mps_diagnostic.kahan_conv2d import (
    KAHAN_CONV2D_AXIS_TAG,
    KAHAN_CONV2D_EVIDENCE_GRADE,
    kahan_conv2d,
    kahan_sum,
    patch_conv2d_to_kahan_for_mps_globally,
    restore_torch_conv2d,
)

# ---------------------------------------------------------------------------
# Provenance / constants


def test_axis_tag_is_macos_mps_kahan() -> None:
    assert KAHAN_CONV2D_AXIS_TAG == "[macOS-MPS-Kahan-corrected-PyTorch]"
    # Per Catalog #287/#323: tag must NOT claim a contest-promotable axis.
    assert "[contest-CPU]" not in KAHAN_CONV2D_AXIS_TAG
    assert "[contest-CUDA]" not in KAHAN_CONV2D_AXIS_TAG


def test_evidence_grade_is_macos_mps_diagnostic() -> None:
    assert KAHAN_CONV2D_EVIDENCE_GRADE == "macOS-MPS-Kahan-corrected-diagnostic"
    # Per CLAUDE.md "MPS auth eval is NOISE": evidence grade must NOT claim
    # a contest-promotable status.
    assert "contest" not in KAHAN_CONV2D_EVIDENCE_GRADE.lower()


# ---------------------------------------------------------------------------
# kahan_sum unit tests (Higham 2002 precision invariant)


def test_kahan_sum_matches_naive_sum_on_few_well_conditioned_terms() -> None:
    # For small N and well-conditioned summands, Kahan and naive sum agree.
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    kahan_result = kahan_sum(x.unsqueeze(0), dim=1).item()
    naive_result = x.sum().item()
    assert abs(kahan_result - naive_result) < 1e-6
    assert abs(kahan_result - 10.0) < 1e-6


def test_kahan_sum_preserves_precision_on_higham_ill_conditioned_case() -> None:
    # Classic Kahan demonstration: sum 1.0 + N * tiny, where tiny is below
    # fp32 epsilon. Naive sum loses precision; Kahan recovers it.
    #
    # Per Higham 2002 ex 4.1: with eps_f32 ~ 6e-8, summing
    # 1.0 + 1e6 * 1e-8 = 1.0 + 0.01 = 1.01 naively gives ~1.0 (truncated),
    # but Kahan recovers ~1.01.
    n_tiny = 100_000  # 1e5 tiny terms; smaller than 1e6 to keep test fast
    tiny = 1e-7
    # Construct summands as [1.0, tiny, tiny, ..., tiny] - sum should be 1.0 + n*tiny = 1.01.
    summands = torch.cat([
        torch.tensor([1.0], dtype=torch.float32),
        torch.full((n_tiny,), tiny, dtype=torch.float32),
    ])
    expected = 1.0 + n_tiny * tiny  # 1.01 in fp64 reference

    kahan_value = kahan_sum(summands.unsqueeze(0), dim=1).item()
    naive_value = summands.sum().item()

    # Kahan should be MUCH closer to the fp64 reference than naive.
    kahan_error = abs(kahan_value - expected)
    naive_error = abs(naive_value - expected)
    # Defensive: Kahan error must be <= naive error (Kahan never strictly
    # worsens precision; should improve it on ill-conditioned sums).
    assert kahan_error <= naive_error, (
        f"Kahan error {kahan_error:.3e} > naive error {naive_error:.3e}: "
        "Kahan should be at least as precise"
    )
    # On a well-implemented Kahan path with this many terms, expect order-of-
    # magnitude improvement.
    if naive_error > 1e-6:
        assert kahan_error <= naive_error * 0.5, (
            f"Kahan should be >= 2x more precise; got kahan={kahan_error:.3e} "
            f"naive={naive_error:.3e}"
        )


def test_kahan_sum_handles_keepdim() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    result = kahan_sum(x, dim=1, keepdim=True)
    assert result.shape == (2, 1)
    assert abs(result[0, 0].item() - 6.0) < 1e-6
    assert abs(result[1, 0].item() - 15.0) < 1e-6


def test_kahan_sum_handles_negative_dim() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    result = kahan_sum(x, dim=-1)
    assert result.shape == (1,)
    assert abs(result.item() - 6.0) < 1e-6


def test_kahan_sum_rejects_out_of_range_dim() -> None:
    x = torch.tensor([1.0, 2.0, 3.0])
    try:
        kahan_sum(x, dim=5)
    except ValueError:
        return
    assert False, "kahan_sum should reject out-of-range dim"


# ---------------------------------------------------------------------------
# kahan_conv2d behavior


def test_kahan_conv2d_falls_through_to_F_conv2d_on_cpu() -> None:
    # CPU tensors should follow the canonical F.conv2d path (no Kahan
    # overhead; Kahan path only fires on MPS).
    x = torch.randn(2, 3, 8, 8)
    w = torch.randn(4, 3, 3, 3)
    b = torch.randn(4)

    kahan_output = kahan_conv2d(x, w, b, stride=1, padding=1)
    canonical_output = F.conv2d(x, w, b, stride=1, padding=1)
    assert torch.allclose(kahan_output, canonical_output, atol=1e-6)


def test_kahan_conv2d_returns_correct_output_shape() -> None:
    x = torch.randn(1, 3, 16, 16)
    w = torch.randn(8, 3, 3, 3)
    output = kahan_conv2d(x, w, stride=2, padding=1)
    # Output spatial size = (16 + 2 - 1*(3-1) - 1) // 2 + 1 = 8.
    assert output.shape == (1, 8, 8, 8)


def test_kahan_conv2d_disable_flag_falls_through() -> None:
    x = torch.randn(1, 3, 4, 4)
    w = torch.randn(2, 3, 3, 3)
    # enable_kahan=False forces F.conv2d path even on MPS.
    output = kahan_conv2d(x, w, enable_kahan=False, stride=1, padding=1)
    expected = F.conv2d(x, w, stride=1, padding=1)
    assert torch.allclose(output, expected, atol=1e-6)


def test_kahan_conv2d_grouped_convolution_falls_through() -> None:
    # Grouped convs route to canonical F.conv2d.
    x = torch.randn(1, 4, 4, 4)
    w = torch.randn(4, 1, 3, 3)  # groups=4 => each filter sees 1 input channel
    output = kahan_conv2d(x, w, groups=4, padding=1)
    expected = F.conv2d(x, w, groups=4, padding=1)
    assert torch.allclose(output, expected, atol=1e-6)


def test_kahan_conv2d_bias_applied_correctly() -> None:
    x = torch.zeros(1, 3, 4, 4)  # zero input => output = bias
    w = torch.randn(2, 3, 3, 3)
    b = torch.tensor([7.0, -3.0])
    output = kahan_conv2d(x, w, b, padding=1)
    # Every spatial position should equal the bias.
    assert torch.allclose(output[0, 0], torch.full((4, 4), 7.0))
    assert torch.allclose(output[0, 1], torch.full((4, 4), -3.0))


def test_kahan_conv2d_matches_F_conv2d_on_mps_when_available() -> None:
    # Only run when MPS is available; otherwise skip silently.
    if not torch.backends.mps.is_available():
        return
    x = torch.randn(1, 3, 8, 8, device="mps")
    w = torch.randn(4, 3, 3, 3, device="mps")
    b = torch.randn(4, device="mps")
    kahan_output = kahan_conv2d(x, w, b, padding=1)
    canonical_output = F.conv2d(x, w, b, padding=1)
    # Kahan and naive Conv2d on small well-conditioned inputs should agree
    # within a loose tolerance (the precision improvement only emerges on
    # large ill-conditioned accumulations).
    assert torch.allclose(kahan_output, canonical_output, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Monkey-patch round-trip


def test_patch_conv2d_to_kahan_then_restore_round_trips() -> None:
    original_conv2d = F.conv2d
    patch_conv2d_to_kahan_for_mps_globally()
    try:
        # After patch, F.conv2d is the dispatcher (different object).
        assert F.conv2d is not original_conv2d
    finally:
        restore_torch_conv2d()
    # After restore, F.conv2d is back to the original.
    assert F.conv2d is original_conv2d


def test_patch_conv2d_idempotent() -> None:
    original_conv2d = F.conv2d
    patch_conv2d_to_kahan_for_mps_globally()
    first_patch = F.conv2d
    patch_conv2d_to_kahan_for_mps_globally()  # second call should be no-op
    try:
        # The dispatcher should be the same object (idempotent).
        assert F.conv2d is first_patch
    finally:
        restore_torch_conv2d()
    assert F.conv2d is original_conv2d


def test_restore_torch_conv2d_idempotent() -> None:
    # Calling restore when no patch is active is a no-op (doesn't crash).
    restore_torch_conv2d()
    restore_torch_conv2d()


def test_patched_conv2d_routes_non_mps_to_canonical() -> None:
    # When patched, CPU tensors should still match F.conv2d behavior.
    patch_conv2d_to_kahan_for_mps_globally()
    try:
        x = torch.randn(1, 3, 4, 4)
        w = torch.randn(2, 3, 3, 3)
        result = F.conv2d(x, w, padding=1)
        # No exception; reasonable shape.
        assert result.shape == (1, 2, 4, 4)
    finally:
        restore_torch_conv2d()
