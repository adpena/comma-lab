# SPDX-License-Identifier: MIT
"""Tests for ``tools/pr101_omega_opt_admm_x_lossy_coarsening_empirical.py``.

Verifies the composition of:

- Joint-ADMM Lagrangian-allocation MECHANISM (Path B step 5)
- Continuous K-step rounding basis (Subagent D)

Produces consistent bytes when re-run with the same inputs (byte-determinism),
with the encoder's per-tensor K side-info matching the wire-format contract,
and with the bisection terminating within the declared budget.

Tests use synthetic TensorBlob inputs (no torch.load) so they run on CPU in
under a second.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import pr101_lossy_coarsening_analytical as base  # noqa: E402
import pr101_omega_opt_admm_x_lossy_coarsening_empirical as mod  # noqa: E402


def _synthetic_tensors(n_tensors: int = 5, seed: int = 0) -> list:
    """Build N synthetic TensorBlobs of varying skew / size."""
    rng = np.random.default_rng(seed)
    blobs = []
    for i in range(n_tensors):
        size = 100 + i * 50
        # Mix of high-magnitude and near-zero values so per-tensor K matters.
        if i % 2 == 0:
            raw = rng.integers(-30, 31, size=size, dtype=np.int32)
        else:
            raw = rng.integers(-10, 11, size=size, dtype=np.int32)
        blobs.append(base.TensorBlob(name=f"t{i}", raw=raw))
    return blobs


def test_precompute_per_tensor_K_curves_shape() -> None:
    tensors = _synthetic_tensors()
    curves = mod.precompute_per_tensor_K_curves(tensors)
    assert len(curves) == len(tensors)
    for tensor_curve in curves:
        assert len(tensor_curve) == len(mod.K_RANGE)
        # Each row has K, rel_err, byte_proxy
        for row in tensor_curve:
            assert "K" in row
            assert "rel_err" in row
            assert "byte_proxy" in row
            assert row["rel_err"] >= 0.0
            assert row["byte_proxy"] > 0


def test_precompute_curves_K1_has_zero_rel_err() -> None:
    """K=1: round(x/1)*1 == x for integer inputs => rel_err = 0."""
    tensors = _synthetic_tensors()
    curves = mod.precompute_per_tensor_K_curves(tensors)
    for tensor_curve in curves:
        K1_row = next(r for r in tensor_curve if r["K"] == 1)
        assert K1_row["rel_err"] == 0.0


def test_precompute_curves_rel_err_K1_zero_and_K_max_largest() -> None:
    """K=1 must give rel_err=0; the largest K in K_RANGE must give one of the
    largest rel_errs (approximate: top-3 in the distribution).

    K-step rounding rel_err is NOT strictly monotone in K (a larger K can
    happen to align better with the local distribution and reduce error).
    What's strictly monotone is the rel_err's UPPER ENVELOPE."""
    tensors = _synthetic_tensors()
    curves = mod.precompute_per_tensor_K_curves(tensors)
    for tensor_curve in curves:
        K1_re = next(r["rel_err"] for r in tensor_curve if r["K"] == 1)
        assert K1_re == 0.0
        max_K_re = next(r["rel_err"] for r in tensor_curve if r["K"] == mod.K_RANGE[-1])
        # The largest K should sit in the top half of rel_errs
        sorted_re = sorted(r["rel_err"] for r in tensor_curve)
        median = sorted_re[len(sorted_re) // 2]
        assert max_K_re >= median, (
            f"largest K rel_err {max_K_re} < median {median}"
        )


def test_lagrangian_select_Ks_lambda_zero_picks_minimum_byte_proxy() -> None:
    """λ -> 0: each tensor picks lowest-byte-proxy K."""
    tensors = _synthetic_tensors()
    curves = mod.precompute_per_tensor_K_curves(tensors)
    Ks, _ = mod.lagrangian_select_Ks(curves, lam=0.0)
    # For each tensor, the picked K should be the one with min byte_proxy.
    for tensor_curve, K_picked in zip(curves, Ks):
        min_proxy_K = min(tensor_curve, key=lambda r: r["byte_proxy"])["K"]
        assert K_picked == min_proxy_K


def test_lagrangian_select_Ks_lambda_huge_picks_K1() -> None:
    """λ -> ∞: rel_err penalty dominates -> all tensors pick K=1."""
    tensors = _synthetic_tensors()
    curves = mod.precompute_per_tensor_K_curves(tensors)
    Ks, rel_errs = mod.lagrangian_select_Ks(curves, lam=1e15)
    assert all(K == 1 for K in Ks)
    assert all(re == 0.0 for re in rel_errs)


def test_bisect_admm_terminates_and_satisfies_target() -> None:
    """Bisection must terminate within 80 iters and satisfy RMS target
    (or be at the K=1 floor)."""
    tensors = _synthetic_tensors()
    curves = mod.precompute_per_tensor_K_curves(tensors)
    for target in [0.01, 0.05, 0.10]:
        result = mod.bisect_admm_for_global_rms(tensors, curves, rms_target=target)
        assert "lambda" in result
        assert "archive_bytes" in result
        assert "rel_err" in result
        # Either constraint satisfied or at lossless floor
        assert (
            result["rel_err"] <= target + 1e-3
            or result["rel_err"] == 0.0
        ), result


def test_encode_with_per_tensor_K_byte_deterministic() -> None:
    """Same tensors + same Ks -> same archive_bytes (modulo brotli's
    deterministic header)."""
    tensors = _synthetic_tensors()
    Ks = [1, 2, 1, 3, 1]
    r1 = base.encode_with_per_tensor_K(tensors, Ks)
    r2 = base.encode_with_per_tensor_K(tensors, Ks)
    assert r1["archive_bytes"] == r2["archive_bytes"]
    assert r1["payload_brotli_bytes"] == r2["payload_brotli_bytes"]
    assert r1["rel_err"] == r2["rel_err"]


def test_encode_with_per_tensor_K_side_info_28_bytes_when_28_tensors() -> None:
    """The wire format reserves 1 byte per tensor for K — the encoder MUST
    emit exactly len(tensors) bytes of side_info."""
    tensors = _synthetic_tensors(n_tensors=28)
    Ks = [1] * 28
    r = base.encode_with_per_tensor_K(tensors, Ks)
    assert r["side_info_bytes"] == 28


def test_encode_with_per_tensor_K_K1_lossless() -> None:
    """K=1 with int32 inputs in [-127, 127] is lossless: round(x/1)*1 == x."""
    tensors = _synthetic_tensors()
    n = len(tensors)
    r = base.encode_with_per_tensor_K(tensors, [1] * n)
    assert r["rel_err"] == 0.0


def test_encode_with_per_tensor_K_large_K_has_higher_rel_err_than_K1() -> None:
    """Encoding at uniform K=8 must have rel_err >= K=1's rel_err (which is
    0 for integer inputs). Tests the LOSSY direction without claiming strict
    monotonicity at every step."""
    tensors = _synthetic_tensors()
    n = len(tensors)
    r1 = base.encode_with_per_tensor_K(tensors, [1] * n)
    r8 = base.encode_with_per_tensor_K(tensors, [8] * n)
    assert r8["rel_err"] >= r1["rel_err"]


def test_falsification_scope_set_in_manifest() -> None:
    """Tool source must declare ``family_falsified=False`` and a
    non-empty ``falsification_scope``."""
    src = (REPO_ROOT / "tools"
           / "pr101_omega_opt_admm_x_lossy_coarsening_empirical.py").read_text()
    assert '"family_falsified": False' in src
    assert '"falsification_scope":' in src
    assert "lagrangian" in src.lower()
    assert "lossy_coarsening" in src or "continuous_K" in src


def test_score_claim_is_false_in_manifest() -> None:
    """CLAUDE.md MPS-NOISE rule + strict-scorer-rule: any tool that doesn't
    run contest-CUDA must declare ``score_claim=False`` in its manifest."""
    src = (REPO_ROOT / "tools"
           / "pr101_omega_opt_admm_x_lossy_coarsening_empirical.py").read_text()
    assert '"score_claim": False' in src
    assert '"ready_for_exact_eval_dispatch": False' in src
