# SPDX-License-Identifier: MIT
"""Tests for the MPS-vs-CUDA engineering corrections surfaced by
``tac.engineered_corrections``.

The three canonical corrections (Catalog #344 equation #2 consumers):

  - ``kahan_summation`` — Kahan-compensated reduction (targets
    ``conv2d_accumulation`` noise source).
  - ``softmax_with_epsilon`` — fp64-pinned softmax (targets
    ``softmax_numerics`` noise source).
  - ``fp32_matmul`` — strict-fp32 matmul accumulation context manager
    (targets ``matmul_fp16`` noise source).

These tests prove the canonical-import-surface contract (the three names
resolve to the same callables as the underlying ``tac.mps_diagnostic.*``
heavy implementations) AND prove the runtime numerics on whichever device
the local box exposes (CPU always; MPS opportunistically; CUDA when
torch.cuda.is_available()).

Per CLAUDE.md "MPS auth eval is NOISE" non-negotiable: every result is
non-promotable; the tests only assert correctness of the correction
mechanism vs the canonical fp64 reference (NOT correctness of any score
claim). Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag":
no docstring or assertion here claims a score reduction; the assertions
verify numerical agreement bounds only.
"""
from __future__ import annotations

import math

import pytest
import torch

from tac.engineered_corrections import (
    KAHAN_CONV2D_AXIS_TAG,
    KAHAN_CONV2D_EVIDENCE_GRADE,
    MPS_BLAS_PREFERENCE_API_AVAILABLE,
    PINNED_SOFTMAX_AXIS_TAG,
    PINNED_SOFTMAX_EVIDENCE_GRADE,
    STRICT_FP32_FLAGS,
    fp32_matmul,
    kahan_summation,
    softmax_with_epsilon,
)


# ── canonical-import-surface contract ───────────────────────────────────────


def test_kahan_summation_resolves_to_mps_diagnostic_kahan_sum() -> None:
    """``tac.engineered_corrections.kahan_summation`` is the same callable
    as ``tac.mps_diagnostic.kahan_conv2d.kahan_sum``.

    This locks in the canonical-import-surface contract so a future refactor
    cannot silently fork the implementation under a new module without the
    Catalog #287 phantom-API guard catching the divergence.
    """
    from tac.mps_diagnostic.kahan_conv2d import kahan_sum

    assert kahan_summation is kahan_sum
    assert kahan_summation.__module__ == "tac.mps_diagnostic.kahan_conv2d"


def test_softmax_with_epsilon_resolves_to_pinned_softmax() -> None:
    """``softmax_with_epsilon`` aliases ``tac.mps_diagnostic.pinned_softmax.pinned_softmax``.

    Same canonical-surface contract as :func:`test_kahan_summation_resolves_to_mps_diagnostic_kahan_sum`.
    """
    from tac.mps_diagnostic.pinned_softmax import pinned_softmax

    assert softmax_with_epsilon is pinned_softmax
    assert softmax_with_epsilon.__module__ == "tac.mps_diagnostic.pinned_softmax"


def test_fp32_matmul_resolves_to_strict_fp32_matmul_accumulation() -> None:
    """``fp32_matmul`` aliases ``tac.mps_diagnostic.fp32_matmul_override.strict_fp32_matmul_accumulation``."""
    from tac.mps_diagnostic.fp32_matmul_override import (
        strict_fp32_matmul_accumulation,
    )

    assert fp32_matmul is strict_fp32_matmul_accumulation


def test_canonical_provenance_tags_match_mps_diagnostic() -> None:
    """Provenance tags re-exported here match the canonical source per
    Catalog #287 / #323.

    Future agents writing audit tooling can rely on the constants being
    importable from EITHER namespace without divergence.
    """
    from tac.mps_diagnostic import kahan_conv2d as _kahan_mod
    from tac.mps_diagnostic import pinned_softmax as _softmax_mod

    assert KAHAN_CONV2D_AXIS_TAG == _kahan_mod.KAHAN_CONV2D_AXIS_TAG
    assert KAHAN_CONV2D_EVIDENCE_GRADE == _kahan_mod.KAHAN_CONV2D_EVIDENCE_GRADE
    assert PINNED_SOFTMAX_AXIS_TAG == _softmax_mod.PINNED_SOFTMAX_AXIS_TAG
    assert PINNED_SOFTMAX_EVIDENCE_GRADE == _softmax_mod.PINNED_SOFTMAX_EVIDENCE_GRADE


def test_strict_fp32_flags_pin_cuda_and_cudnn_tf32() -> None:
    """The strict-fp32 flag tuple covers the two canonical TF32 toggles."""
    flag_names = {(mod, attr) for (mod, attr) in STRICT_FP32_FLAGS}
    assert ("torch.backends.cuda.matmul", "allow_tf32") in flag_names
    assert ("torch.backends.cudnn", "allow_tf32") in flag_names


# ── canonical equation #2 cross-reference ───────────────────────────────────


def test_canonical_equation_2_lists_engineered_corrections_consumers() -> None:
    """Equation #2 (``mps_drift_architecture_class_dependent_v1``)'s
    ``canonical_consumers`` tuple cites the three corrections per Catalog #344.

    This locks in the producer/consumer audit invariant (Catalog #265
    sister pattern at the canonical-equations surface) so the equation
    registry's invariant ``__post_init__`` keeps the corrections wired.
    """
    from tac.canonical_equations.builtins import (
        build_mps_drift_architecture_class_dependent_v1,
    )

    equation = build_mps_drift_architecture_class_dependent_v1()
    consumers = set(equation.canonical_consumers)
    assert "tac.engineered_corrections.kahan_summation" in consumers
    assert "tac.engineered_corrections.softmax_with_epsilon" in consumers
    assert "tac.engineered_corrections.fp32_matmul" in consumers


# ── kahan_summation numerical correctness ───────────────────────────────────


def test_kahan_summation_matches_torch_sum_on_well_conditioned_input() -> None:
    """On a benign 1-D fp32 tensor (no catastrophic cancellation), Kahan
    summation agrees with ``torch.sum`` to within machine epsilon * N.
    """
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.float32)
    kahan = kahan_summation(x, dim=0)
    naive = x.sum()
    assert torch.allclose(kahan, naive, atol=1e-4, rtol=1e-5)


def test_kahan_summation_beats_naive_on_catastrophic_cancellation() -> None:
    """Build a worst-case summand sequence whose naive fp32 sum drifts from
    the fp64 reference; verify Kahan stays closer to the reference.

    The construction: large positive base + many small ones whose total
    equals a third positive contribution. Naive fp32 loses the small
    contributions because each one rounds to zero against the large base.
    """
    large = 1.0e7
    small_val = 1.0
    n_small = 10_000
    summands_fp32 = torch.cat(
        [
            torch.tensor([large], dtype=torch.float32),
            torch.full((n_small,), small_val, dtype=torch.float32),
            torch.tensor([-large], dtype=torch.float32),
        ]
    )
    summands_fp64 = summands_fp32.to(dtype=torch.float64)

    naive_fp32 = summands_fp32.sum()
    kahan_fp32 = kahan_summation(summands_fp32, dim=0)
    reference_fp64 = summands_fp64.sum()

    naive_error = float(abs(naive_fp32 - reference_fp64))
    kahan_error = float(abs(kahan_fp32 - reference_fp64))
    # Kahan should be strictly better or equivalent; reference is n_small=1e4
    # so naive error is bounded ~ O(eps * 1e7) ~ 1.0 while Kahan error ~ eps.
    assert kahan_error <= naive_error
    # Reference result is exactly n_small=10_000 (large + (-large) cancel).
    assert math.isclose(float(kahan_fp32), float(reference_fp64), abs_tol=1e-1)


def test_kahan_summation_preserves_input_dtype() -> None:
    """Output dtype equals input dtype (per the docstring contract)."""
    x = torch.randn(64, dtype=torch.float32)
    assert kahan_summation(x, dim=0).dtype == torch.float32

    x64 = torch.randn(64, dtype=torch.float64)
    assert kahan_summation(x64, dim=0).dtype == torch.float64


def test_kahan_summation_supports_keepdim() -> None:
    """``keepdim=True`` reinserts the reduced axis as size-1."""
    x = torch.randn(4, 8, dtype=torch.float32)
    out = kahan_summation(x, dim=1, keepdim=True)
    assert out.shape == (4, 1)


def test_kahan_summation_supports_negative_dim() -> None:
    """``dim=-1`` reduces the last axis (canonical PyTorch convention)."""
    x = torch.randn(3, 5, dtype=torch.float32)
    out = kahan_summation(x, dim=-1)
    assert out.shape == (3,)


def test_kahan_summation_rejects_out_of_range_dim() -> None:
    """Invalid ``dim`` raises ``ValueError`` (per helper contract)."""
    x = torch.randn(8, dtype=torch.float32)
    with pytest.raises(ValueError, match="out of range"):
        kahan_summation(x, dim=2)


# ── softmax_with_epsilon numerical correctness ──────────────────────────────


def test_softmax_with_epsilon_fallback_matches_torch_softmax_on_cpu() -> None:
    """On non-MPS devices (CPU here), ``softmax_with_epsilon`` falls
    through to ``F.softmax`` and matches its output exactly.
    """
    import torch.nn.functional as F

    torch.manual_seed(0)
    logits = torch.randn(4, 5, dtype=torch.float32)
    pinned = softmax_with_epsilon(logits, dim=-1)
    naive = F.softmax(logits, dim=-1)
    assert torch.allclose(pinned, naive, atol=1e-7, rtol=1e-6)


def test_softmax_with_epsilon_sums_to_one_per_row() -> None:
    """Softmax rows sum to 1 within fp32 rounding tolerance."""
    logits = torch.randn(8, 16, dtype=torch.float32)
    out = softmax_with_epsilon(logits, dim=-1)
    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5)


def test_softmax_with_epsilon_preserves_input_dtype() -> None:
    """Output dtype matches input dtype after fp64 stabilization + demotion."""
    logits = torch.randn(3, 4, dtype=torch.float32)
    assert softmax_with_epsilon(logits, dim=-1).dtype == torch.float32


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available on this machine",
)
def test_softmax_with_epsilon_on_mps_matches_fp64_reference() -> None:
    """On MPS, ``softmax_with_epsilon`` matches the fp64 CPU reference
    within fp32 machine epsilon * O(N) — this is the canonical proof
    that the fp64 stabilization closes the Metal-vs-cuDNN epsilon gap
    relative to the canonical reference.
    """
    torch.manual_seed(0)
    logits_cpu = torch.randn(8, 5, dtype=torch.float32)
    logits_mps = logits_cpu.to(device="mps")

    pinned_mps = softmax_with_epsilon(logits_mps, dim=-1).cpu()
    reference_fp64 = torch.nn.functional.softmax(
        logits_cpu.to(dtype=torch.float64), dim=-1
    )

    assert torch.allclose(
        pinned_mps.to(dtype=torch.float64),
        reference_fp64,
        atol=1e-5,
        rtol=1e-5,
    )


# ── fp32_matmul context-manager correctness ─────────────────────────────────


def test_fp32_matmul_disables_and_restores_cuda_tf32() -> None:
    """The context manager toggles ``torch.backends.cuda.matmul.allow_tf32``
    to False inside the block AND restores the prior value on exit.

    Runs on every box; when CUDA backend attrs are missing the helper
    records ``None`` and the test verifies idempotent restoration.
    """
    cuda_matmul = getattr(torch.backends.cuda, "matmul", None)
    if cuda_matmul is None or not hasattr(cuda_matmul, "allow_tf32"):
        pytest.skip("torch.backends.cuda.matmul.allow_tf32 not present on this build")

    prior = cuda_matmul.allow_tf32
    try:
        with fp32_matmul():
            assert cuda_matmul.allow_tf32 is False
        # Restored on exit.
        assert cuda_matmul.allow_tf32 == prior
    finally:
        cuda_matmul.allow_tf32 = prior


def test_fp32_matmul_disables_and_restores_cudnn_tf32() -> None:
    """Sister assertion for ``torch.backends.cudnn.allow_tf32``."""
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is None or not hasattr(cudnn, "allow_tf32"):
        pytest.skip("torch.backends.cudnn.allow_tf32 not present on this build")

    prior = cudnn.allow_tf32
    try:
        with fp32_matmul():
            assert cudnn.allow_tf32 is False
        assert cudnn.allow_tf32 == prior
    finally:
        cudnn.allow_tf32 = prior


def test_fp32_matmul_records_mps_blas_preference_api_availability() -> None:
    """The runtime probe ``MPS_BLAS_PREFERENCE_API_AVAILABLE`` matches the
    actual presence of ``torch.backends.mps.preferred_blas_library`` per
    Catalog #229 PV.

    Per the slot 9 formalization the attribute does NOT exist in torch
    2.11; the probe must surface the absence rather than silently degrade.
    """
    expected = hasattr(torch.backends, "mps") and hasattr(
        torch.backends.mps, "preferred_blas_library"
    )
    assert MPS_BLAS_PREFERENCE_API_AVAILABLE is expected


def test_fp32_matmul_is_reentrant_safe() -> None:
    """Nested context managers restore prior state in LIFO order."""
    cuda_matmul = getattr(torch.backends.cuda, "matmul", None)
    if cuda_matmul is None or not hasattr(cuda_matmul, "allow_tf32"):
        pytest.skip("torch.backends.cuda.matmul.allow_tf32 not present on this build")

    prior = cuda_matmul.allow_tf32
    try:
        cuda_matmul.allow_tf32 = True
        with fp32_matmul():
            assert cuda_matmul.allow_tf32 is False
            with fp32_matmul():
                assert cuda_matmul.allow_tf32 is False
            # Inner exit restores to outer-block False (inner observed False).
            assert cuda_matmul.allow_tf32 is False
        # Outer exit restores to pre-block True.
        assert cuda_matmul.allow_tf32 is True
    finally:
        cuda_matmul.allow_tf32 = prior


# ── canonical-equation domain-of-validity audit ─────────────────────────────


def test_canonical_equation_2_domain_lists_three_noise_sources() -> None:
    """Equation #2's ``domain_of_validity`` lists exactly the three noise
    sources the corrections target. Locks the wire-in at the registry
    surface so a future agent cannot add a fourth correction without also
    extending the equation's domain.
    """
    from tac.canonical_equations.builtins import (
        build_mps_drift_architecture_class_dependent_v1,
    )

    equation = build_mps_drift_architecture_class_dependent_v1()
    sources = set(equation.domain_of_validity["noise_sources_modeled"])
    assert sources == {
        "conv2d_accumulation",
        "softmax_numerics",
        "matmul_fp16",
    }
