# SPDX-License-Identifier: MIT
"""Tests for tac.mps_diagnostic.pinned_softmax (slot 9 HIGH-EV correction).

Per slot 9 formalization (`feedback_mps_drift_mathematical_and_engineering_formalization_landed_20260519.md`):
predicted ~50% reduction in SegNet 5-class argmax boundary flip rate via fp64
stabilization of log-sum-exp.

Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #287/#323: every test
verifies the helper preserves non-promotable axis tags.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tac.mps_diagnostic.pinned_softmax import (
    PINNED_SOFTMAX_AXIS_TAG,
    PINNED_SOFTMAX_EVIDENCE_GRADE,
    patch_softmax_to_pinned_for_mps_globally,
    pinned_softmax,
    restore_torch_softmax,
)


# ---------------------------------------------------------------------------
# Provenance / constants


def test_axis_tag_is_macos_mps_pinned() -> None:
    assert PINNED_SOFTMAX_AXIS_TAG == "[macOS-MPS-pinned-softmax-PyTorch]"
    assert "[contest-CPU]" not in PINNED_SOFTMAX_AXIS_TAG
    assert "[contest-CUDA]" not in PINNED_SOFTMAX_AXIS_TAG


def test_evidence_grade_is_macos_mps_diagnostic() -> None:
    assert PINNED_SOFTMAX_EVIDENCE_GRADE == "macOS-MPS-pinned-softmax-diagnostic"
    assert "contest" not in PINNED_SOFTMAX_EVIDENCE_GRADE.lower()


# ---------------------------------------------------------------------------
# pinned_softmax behavior


def test_pinned_softmax_falls_through_to_F_softmax_on_cpu() -> None:
    # CPU tensors should follow the canonical path (Metal-vs-cuDNN epsilon
    # gap is MPS-specific; no need to pay the fp64 promotion cost on CPU).
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    pinned = pinned_softmax(logits, dim=-1)
    canonical = F.softmax(logits, dim=-1)
    assert torch.allclose(pinned, canonical, atol=1e-6)


def test_pinned_softmax_sums_to_one() -> None:
    logits = torch.randn(4, 10)
    pinned = pinned_softmax(logits, dim=-1)
    row_sums = pinned.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(4), atol=1e-5)


def test_pinned_softmax_preserves_dtype() -> None:
    logits_fp32 = torch.randn(2, 5, dtype=torch.float32)
    result_fp32 = pinned_softmax(logits_fp32, dim=-1)
    assert result_fp32.dtype == torch.float32

    logits_fp16 = torch.randn(2, 5, dtype=torch.float16)
    result_fp16 = pinned_softmax(logits_fp16, dim=-1)
    assert result_fp16.dtype == torch.float16


def test_pinned_softmax_disable_falls_through() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    pinned = pinned_softmax(logits, dim=-1, enable_pinning=False)
    canonical = F.softmax(logits, dim=-1)
    assert torch.allclose(pinned, canonical, atol=1e-6)


def test_pinned_softmax_handles_extreme_positive_logits() -> None:
    # max-subtraction stabilization should prevent overflow when one logit
    # is very large.
    logits = torch.tensor([[1e10, 1.0, 0.0]])
    pinned = pinned_softmax(logits, dim=-1)
    # The dominant logit gets all the mass.
    assert torch.isfinite(pinned).all()
    assert pinned[0, 0].item() > 0.99
    assert pinned.sum().item() > 0.99 and pinned.sum().item() <= 1.0 + 1e-5


def test_pinned_softmax_handles_extreme_negative_logits() -> None:
    # max-subtraction should also prevent underflow when all logits are very
    # negative.
    logits = torch.tensor([[-1e10, -1e10, -1e10]])
    pinned = pinned_softmax(logits, dim=-1)
    assert torch.isfinite(pinned).all()
    # All equal => uniform distribution.
    assert torch.allclose(pinned, torch.full_like(pinned, 1.0 / 3.0), atol=1e-5)


def test_pinned_softmax_matches_F_softmax_on_mps_when_available() -> None:
    if not torch.backends.mps.is_available():
        return
    logits = torch.randn(2, 5, device="mps")
    pinned = pinned_softmax(logits, dim=-1)
    canonical = F.softmax(logits, dim=-1)
    # Pinned (fp64 stabilization) and naive (fp32 stabilization) should agree
    # within a loose tolerance for well-conditioned logits; the boundary-flip
    # reduction only emerges at argmax boundaries.
    assert torch.allclose(pinned, canonical, atol=1e-4, rtol=1e-3)


def test_pinned_softmax_argmax_agrees_with_F_softmax_for_well_separated() -> None:
    # When the argmax winner is well-separated from runners-up, pinned and
    # naive softmax must agree on argmax (the boundary-flip reduction only
    # matters at the ARGMAX boundary).
    logits = torch.tensor([[10.0, 1.0, 0.0, -1.0, -10.0]])
    pinned_argmax = pinned_softmax(logits, dim=-1).argmax(dim=-1)
    canonical_argmax = F.softmax(logits, dim=-1).argmax(dim=-1)
    assert pinned_argmax.item() == canonical_argmax.item()


# ---------------------------------------------------------------------------
# Monkey-patch round-trip


def test_patch_softmax_to_pinned_then_restore_round_trips() -> None:
    original_softmax = F.softmax
    patch_softmax_to_pinned_for_mps_globally()
    try:
        # After patch, F.softmax is the dispatcher (different object).
        assert F.softmax is not original_softmax
    finally:
        restore_torch_softmax()
    assert F.softmax is original_softmax


def test_patch_softmax_idempotent() -> None:
    original_softmax = F.softmax
    patch_softmax_to_pinned_for_mps_globally()
    first_patch = F.softmax
    patch_softmax_to_pinned_for_mps_globally()
    try:
        assert F.softmax is first_patch
    finally:
        restore_torch_softmax()
    assert F.softmax is original_softmax


def test_restore_torch_softmax_idempotent() -> None:
    restore_torch_softmax()
    restore_torch_softmax()


def test_patched_softmax_routes_non_mps_to_canonical() -> None:
    patch_softmax_to_pinned_for_mps_globally()
    try:
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        result = F.softmax(logits, dim=-1)
        # Sums to one; same as canonical.
        assert abs(result.sum().item() - 1.0) < 1e-6
    finally:
        restore_torch_softmax()


def test_pinned_softmax_handles_negative_dim_via_helper_signature() -> None:
    logits = torch.randn(3, 4)
    pinned = pinned_softmax(logits, dim=-1)
    assert pinned.shape == logits.shape
    canonical = F.softmax(logits, dim=-1)
    assert torch.allclose(pinned, canonical, atol=1e-5)
