# SPDX-License-Identifier: MIT
"""Unit tests for PR106 cross-substrate Lagrangian per-tensor allocation port.

Verifies the Path B step 5/6 + UNIWARD mechanism transfers from PR101 to PR106:

1. zigzag uint8 ↔ signed int8 roundtrip (PR106 wire-format primitive)
2. ``_find_best_K_per_tensor`` monotonicity in budget
3. UNIWARD weights are 1/(variance + eps) (inversely proportional to spread)
4. Real PR106 archive K=1 lossless roundtrip reproduces published bytes exactly
   (170,278 decoder_brotli; 186,239 archive)
5. K>1 reduces archive bytes and produces nonzero rel_err

These tests stay CPU-only and do not invoke any scorer; the cross-substrate
result they back is byte-anchor only, never a score claim, per CLAUDE.md
``forbidden_CPU_MPS_derived_dispatch_readiness_flag``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))


@pytest.fixture
def pr106_tools_module():
    import pr106_omega_opt_lagrangian_per_tensor_allocation_empirical as mod
    return mod


def test_zigzag_roundtrip(pr106_tools_module):
    """Zigzag-uint8 encode/decode must roundtrip every signed int8 value."""
    orig = np.arange(-127, 128, dtype=np.int32)
    zz_bytes = pr106_tools_module._i8_to_zz_u8(orig)
    back = pr106_tools_module._zz_u8_to_i8(zz_bytes)
    assert (back == orig).all(), "zigzag roundtrip broke for full int8 range"


def test_zigzag_extremes(pr106_tools_module):
    """Edge values: -127, -1, 0, 1, 127."""
    orig = np.array([-127, -1, 0, 1, 127], dtype=np.int32)
    zz = pr106_tools_module._i8_to_zz_u8(orig)
    back = pr106_tools_module._zz_u8_to_i8(zz)
    assert (back == orig).all()


def test_find_best_K_monotone_in_budget(pr106_tools_module):
    """Larger budget must permit at least as large K."""
    sym = np.array([10, 20, 30, 40, -50, -60], dtype=np.int32)
    K_lo, _ = pr106_tools_module._find_best_K_per_tensor(sym, 0.001)
    K_hi, _ = pr106_tools_module._find_best_K_per_tensor(sym, 0.5)
    assert K_hi >= K_lo


def test_find_best_K_zero_tensor(pr106_tools_module):
    """All-zero tensor must return K=1 with zero rel_err."""
    sym = np.zeros(10, dtype=np.int32)
    K, re = pr106_tools_module._find_best_K_per_tensor(sym, 0.05)
    assert K == 1
    assert re == 0.0


def test_uniward_weights_inverse_variance(pr106_tools_module):
    """Low-variance tensor must receive higher (more constraining) weight."""
    mod = pr106_tools_module
    high_var = np.array(
        [100, -100, 50, -50, 80, -80, 60, -60, 40, -40], dtype=np.int32
    )
    low_var = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=np.int32)
    tensors = [
        mod.PR106TensorBlob("hi_var", (10,), high_var, b"\x00" * 4),
        mod.PR106TensorBlob("lo_var", (10,), low_var, b"\x00" * 4),
    ]
    weights = mod._compute_uniward_weights(tensors)
    assert weights[1] > weights[0]
    # ratio should reflect variance ratio (inverse) closely
    assert weights[1] / weights[0] > 100.0


def test_real_pr106_K1_lossless_roundtrip(pr106_tools_module):
    """K=1 baseline must reproduce PR106's published decoder/archive bytes
    exactly (proves the wire-format port is byte-correct)."""
    mod = pr106_tools_module
    archive = (
        REPO_ROOT
        / "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip"
    )
    if not archive.is_file():
        pytest.skip(f"PR106 archive not present at {archive}")
    tensors = mod.collect_pr106_tensors(archive)
    assert len(tensors) == 28
    assert sum(tb.raw_i8.size for tb in tensors) == 228_958

    result = mod._encode_decoder_brotli_with_per_tensor_K(
        tensors, [1] * len(tensors)
    )
    assert result["decoder_brotli_bytes"] == mod.PR106_DECODER_BROTLI_BASELINE_BYTES
    assert result["archive_bytes"] == mod.PR106_ARCHIVE_BASELINE_BYTES
    assert result["rel_err"] == 0.0


def test_real_pr106_K_gt_1_reduces_archive(pr106_tools_module):
    """K=2 globally must yield smaller archive AND nonzero rel_err."""
    mod = pr106_tools_module
    archive = (
        REPO_ROOT
        / "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip"
    )
    if not archive.is_file():
        pytest.skip(f"PR106 archive not present at {archive}")
    tensors = mod.collect_pr106_tensors(archive)
    base = mod._encode_decoder_brotli_with_per_tensor_K(
        tensors, [1] * len(tensors)
    )
    k2 = mod._encode_decoder_brotli_with_per_tensor_K(
        tensors, [2] * len(tensors)
    )
    assert k2["archive_bytes"] < base["archive_bytes"]
    assert k2["rel_err"] > 0.0


def test_archive_overhead_constant_consistent(pr106_tools_module):
    """Validate the archive-overhead arithmetic: archive == decoder + overhead."""
    mod = pr106_tools_module
    expected = (
        mod.PR106_DECODER_BROTLI_BASELINE_BYTES + mod.PR106_ARCHIVE_OVERHEAD_BYTES
    )
    assert expected == mod.PR106_ARCHIVE_BASELINE_BYTES, (
        f"PR106 archive arithmetic broken: {expected} != "
        f"{mod.PR106_ARCHIVE_BASELINE_BYTES}"
    )


def test_dispatch_blockers_present(pr106_tools_module):
    """Every CPU-prep tool must declare dispatch_blockers (CLAUDE.md
    forbidden_CPU_MPS_derived_dispatch_readiness_flag)."""
    mod = pr106_tools_module
    assert "missing_exact_cuda_auth_eval" in mod.DISPATCH_BLOCKERS
    assert (
        "no_runtime_dequantize_path_built_for_modified_decoder"
        in mod.DISPATCH_BLOCKERS
    )
    assert "byte_rel_err_proxy_only_no_score_test" in mod.DISPATCH_BLOCKERS
    assert mod.EVIDENCE_GRADE.startswith("[CPU-prep")
