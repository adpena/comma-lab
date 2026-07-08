# SPDX-License-Identifier: MIT
"""Cross-process MLX-GPU bit-identity guarantees (task #348, 2026-07-07).

POSITIVE guarantees (these are what the witness GPU-determinism path relies on):
  * the fused-R Metal VJP (``make_fused_r_roundtrip`` — the ``--fused-r-kernel``
    lever) is cross-process BIT-IDENTICAL on GPU (fixed-order transpose matmuls,
    no atomics). This is the op that replaced the nondeterministic gather-based
    reference-R backward and made the full witness trainer 0/28-diverged
    (measured N=10, ledger deterministic_gpu_accum_348_20260707).
  * core GEMM forwards are cross-process bit-identical (kernel-selection sanity).

MECHANISM documentation (non-flaky): the gather-based reference-R bicubic-UP
backward (``r_up_grad``) and duplicate-index scatter-add are the measured
nondeterminism sources. If a future MLX release makes them deterministic these
tests WARN (news, not failure) rather than fail.

All verdicts here are [macOS-MLX research-signal] bit-identity facts — never
score claims. Tests skip cleanly off-Metal (CI / Linux).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "tools"))


def _mlx_gpu_available() -> bool:
    try:
        import mlx.core as mx

        return mx.metal.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _mlx_gpu_available(), reason="MLX Metal GPU not available"
)


def _cell(op: str, n: int = 3) -> dict:
    from mlx_gpu_determinism_probe import probe_cell

    r = probe_cell(op, device="gpu", n=n)
    assert "error" not in r, f"probe child failed: {r.get('error')}"
    return r


def test_fused_r_vjp_cross_process_bit_identical() -> None:
    """The determinism guarantee the witness GPU path relies on (positive)."""
    r = _cell("fused_r_vjp")
    assert r["cross_process_identical"] is True, r
    assert r["in_process_identical_all"] is True, r


def test_fused_r_forward_cross_process_bit_identical() -> None:
    r = _cell("fused_r_forward")
    assert r["cross_process_identical"] is True, r


def test_gemm_cross_process_bit_identical() -> None:
    """Core GEMM (incl. the split-K-shaped huge-K case) is deterministic."""
    for op in ("matmul_square", "matmul_bigK"):
        r = _cell(op)
        assert r["cross_process_identical"] is True, r


def test_reference_r_up_backward_mechanism_documented() -> None:
    """The MEASURED nondeterminism source: gather-based bicubic-UP backward.

    Non-flaky by design: nondeterminism is the DOCUMENTED current state; if an
    MLX upgrade makes it deterministic that is NEWS (the CPU-locked bit-exact
    discipline could relax further) — warn, never fail.
    """
    r = _cell("r_up_grad")
    if r["cross_process_identical"]:
        warnings.warn(
            "reference-R bicubic-UP backward has BECOME cross-process deterministic "
            "(MLX upgrade?). Re-run tools/mlx_gpu_determinism_probe.py and revisit "
            "the CPU-locked bit-exact discipline (memory L70 / ledger "
            "deterministic_gpu_accum_348_20260707).",
            stacklevel=1,
        )


def test_scatter_add_dup_mechanism_documented() -> None:
    r = _cell("scatter_add_dup")
    if r["cross_process_identical"]:
        warnings.warn(
            "duplicate-index scatter-add has BECOME deterministic on Metal "
            "(MLX upgrade?). Revisit the #348 localization table.",
            stacklevel=1,
        )
