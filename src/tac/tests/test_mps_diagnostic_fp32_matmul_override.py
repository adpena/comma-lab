# SPDX-License-Identifier: MIT
"""Tests for tac.mps_diagnostic.fp32_matmul_override (slot 9 MEDIUM-EV correction).

Per slot 9 formalization (`feedback_mps_drift_mathematical_and_engineering_formalization_landed_20260519.md`):
predicted 2x drift reduction from TF32 vs IEEE strict fp32 mantissa gap.

Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #229 PV: torch 2.11.0 does
NOT expose `torch.backends.mps.preferred_blas_library`; the helper records the
structural absence in `MPS_BLAS_PREFERENCE_API_AVAILABLE` so callers can audit.
"""
from __future__ import annotations

import torch

from tac.mps_diagnostic.fp32_matmul_override import (
    MPS_BLAS_PREFERENCE_API_AVAILABLE,
    STRICT_FP32_FLAGS,
    enable_fp32_matmul_accumulation_strict,
    restore_fp32_matmul_accumulation_state,
    strict_fp32_matmul_accumulation,
)


def test_strict_fp32_flags_constant_is_tuple_of_pairs() -> None:
    assert isinstance(STRICT_FP32_FLAGS, tuple)
    assert len(STRICT_FP32_FLAGS) >= 2
    for entry in STRICT_FP32_FLAGS:
        assert isinstance(entry, tuple) and len(entry) == 2
        assert isinstance(entry[0], str) and isinstance(entry[1], str)


def test_strict_fp32_flags_includes_cuda_matmul_and_cudnn() -> None:
    pairs = set(STRICT_FP32_FLAGS)
    assert ("torch.backends.cuda.matmul", "allow_tf32") in pairs
    assert ("torch.backends.cudnn", "allow_tf32") in pairs


def test_mps_blas_preference_api_available_constant_is_bool() -> None:
    # Per Catalog #229 PV: torch 2.11.0 does NOT expose preferred_blas_library;
    # the helper records the structural absence so callers can audit.
    assert isinstance(MPS_BLAS_PREFERENCE_API_AVAILABLE, bool)


def test_enable_returns_prior_state_dict() -> None:
    prior = enable_fp32_matmul_accumulation_strict()
    try:
        assert isinstance(prior, dict)
        # Every entry must be either bool/str (real prior) or None (attr absent).
        for key, value in prior.items():
            assert isinstance(key, str)
            assert value is None or isinstance(value, (bool, str))
    finally:
        restore_fp32_matmul_accumulation_state(prior)


def test_enable_disables_cuda_tf32_when_available() -> None:
    # cuda.matmul.allow_tf32 attribute should exist on torch 2.11.0 even on
    # CPU-only builds (PyTorch defines the attribute regardless of CUDA build).
    if not hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        # Defensive skip; cannot test what isn't present.
        return
    prior = enable_fp32_matmul_accumulation_strict()
    try:
        # After enable, flag should be False.
        assert torch.backends.cuda.matmul.allow_tf32 is False
    finally:
        restore_fp32_matmul_accumulation_state(prior)
    # After restore, flag is back to prior value.
    assert torch.backends.cuda.matmul.allow_tf32 == prior["torch.backends.cuda.matmul.allow_tf32"]


def test_enable_disables_cudnn_tf32_when_available() -> None:
    if not hasattr(torch.backends.cudnn, "allow_tf32"):
        return
    prior = enable_fp32_matmul_accumulation_strict()
    try:
        assert torch.backends.cudnn.allow_tf32 is False
    finally:
        restore_fp32_matmul_accumulation_state(prior)
    assert torch.backends.cudnn.allow_tf32 == prior["torch.backends.cudnn.allow_tf32"]


def test_enable_records_none_for_absent_mps_blas_preference() -> None:
    # Per Catalog #229 PV: torch 2.11.0 does not expose this attribute, so
    # the helper records None in prior_state so callers can audit the
    # structural skip. (If a future torch version exposes the API, this test
    # would need to be updated.)
    prior = enable_fp32_matmul_accumulation_strict()
    try:
        if not MPS_BLAS_PREFERENCE_API_AVAILABLE:
            assert prior["torch.backends.mps.preferred_blas_library"] is None
    finally:
        restore_fp32_matmul_accumulation_state(prior)


def test_restore_round_trips_cuda_allow_tf32() -> None:
    if not hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        return
    # Pin to a known prior state.
    original_value = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        prior = enable_fp32_matmul_accumulation_strict()
        assert torch.backends.cuda.matmul.allow_tf32 is False
        restore_fp32_matmul_accumulation_state(prior)
        assert torch.backends.cuda.matmul.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_value


def test_restore_skips_none_entries() -> None:
    # restore_fp32_matmul_accumulation_state should not crash on a None entry.
    prior = {
        "torch.backends.mps.preferred_blas_library": None,
        "torch.backends.cuda.matmul.allow_tf32": True,
    }
    # Save current state to restore after.
    current_cuda = torch.backends.cuda.matmul.allow_tf32 if hasattr(torch.backends.cuda.matmul, "allow_tf32") else None
    try:
        restore_fp32_matmul_accumulation_state(prior)
        # No exception raised.
    finally:
        if current_cuda is not None:
            torch.backends.cuda.matmul.allow_tf32 = current_cuda


def test_context_manager_enables_then_restores() -> None:
    if not hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        return
    original_value = torch.backends.cuda.matmul.allow_tf32
    # Force a known starting value.
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with strict_fp32_matmul_accumulation():
            assert torch.backends.cuda.matmul.allow_tf32 is False
        # On exit, restored.
        assert torch.backends.cuda.matmul.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_value


def test_context_manager_restores_on_exception() -> None:
    if not hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        return
    original_value = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        try:
            with strict_fp32_matmul_accumulation():
                assert torch.backends.cuda.matmul.allow_tf32 is False
                raise RuntimeError("simulated failure inside block")
        except RuntimeError:
            pass
        # State restored despite exception.
        assert torch.backends.cuda.matmul.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_value


def test_include_mps_blas_preference_false_skips_mps_key() -> None:
    prior = enable_fp32_matmul_accumulation_strict(include_mps_blas_preference=False)
    try:
        assert "torch.backends.mps.preferred_blas_library" not in prior
    finally:
        restore_fp32_matmul_accumulation_state(prior)
