"""Tests for the Hessian-preconditioned damped-Newton head-offset option (Plus-Gourdon &
Nielsen, arXiv 2606.09077) on :func:`damped_newton_ot_offsets`.

Verifies: (1) the legacy ``precondition=False`` step is BYTE-IDENTICAL to the pre-2026-07-10
pinv path (parity contract); (2) preconditioning reaches the SAME fixed point ``b*`` on a
well-conditioned problem (correctness — it changes numerics, not the objective); (3) a
positive control where the AVERAGED Hessian is genuinely ill-conditioned exercises the
preconditioned path and it still converges; (4) the structure-tensor ``cond_gate`` falls
through to the legacy path byte-identically; (5) the range condition-number sensor.

Run: ``.venv/bin/python -m pytest src/tac/boundary_math/tests/test_hessian_precond_ot.py``
"""
from __future__ import annotations

import numpy as np

from tac.boundary_math.laguerre_logit_offset import (
    _newton_step_from_cov,
    damped_newton_ot_offsets,
    softmax_cov_condition_number,
)

_RNG = np.random.default_rng(20260710)


def _well_conditioned(n: int = 5000, k: int = 5):
    phi = _RNG.standard_normal((n, k))
    tgt = np.array([0.22, 0.02, 0.49, 0.02, 0.25])
    return phi, tgt


def test_newton_step_legacy_is_byte_identical_to_pinv() -> None:
    """precondition=False MUST reproduce ``taus * pinv(cov, rcond) @ g`` exactly (parity)."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        s = rng.random((300, 5))
        s = s / s.sum(axis=1, keepdims=True)
        m = s.mean(axis=0)
        cov = np.diag(m) - (s.T @ s) / s.shape[0]
        g = rng.standard_normal(5)
        g = g - g.mean()
        taus, rcond = 1.0, 1e-10
        step, cond = _newton_step_from_cov(cov, g, taus, precondition=False,
                                           rcond=rcond, eps_rel=1e-9, cond_gate=None)
        ref = taus * (np.linalg.pinv(cov, rcond=rcond) @ g)
        assert np.array_equal(step, ref), "legacy Newton step drifted from the pinv reference"
        assert np.isfinite(cond)


def test_precond_reaches_same_fixed_point_well_conditioned() -> None:
    phi, tgt = _well_conditioned()
    b_leg, i_leg = damped_newton_ot_offsets(phi, tgt, precondition=False)
    b_pre, i_pre = damped_newton_ot_offsets(phi, tgt, precondition=True)
    assert i_leg["converged"] == 1.0 and i_pre["converged"] == 1.0
    # SAME objective, SAME fixed point (preconditioning changes only per-step numerics)
    assert np.max(np.abs(b_leg - b_pre)) < 1e-8
    assert i_pre["preconditioned"] == 1.0
    assert np.isfinite(i_pre["cond_number"]) and i_pre["cond_number"] > 0.0


def test_cond_gate_fallthrough_is_byte_identical() -> None:
    """A huge cond_gate => the eigendecomp never pays => byte-identical to legacy."""
    phi, tgt = _well_conditioned()
    b_leg, _ = damped_newton_ot_offsets(phi, tgt, precondition=False)
    b_gate, _ = damped_newton_ot_offsets(phi, tgt, precondition=True, precond_cond_gate=1e18)
    assert np.array_equal(b_leg, b_gate)


def _ill_conditioned_softmax_cov(evals_range) -> np.ndarray:
    """A valid softmax-shaped PSD matrix: all-ones gauge nullspace (eigenvalue 0) plus a
    prescribed ill-conditioned range spectrum on the orthogonal complement."""
    k = len(evals_range) + 1
    ones = np.ones(k) / np.sqrt(k)
    # orthonormal basis with first column = all-ones/sqrt(k)
    a = np.eye(k)
    a[:, 0] = ones
    q, _ = np.linalg.qr(a)
    if np.dot(q[:, 0], ones) < 0:
        q[:, 0] = -q[:, 0]
    evals = np.concatenate([[0.0], np.asarray(evals_range, dtype=np.float64)])
    cov = (q * evals) @ q.T
    return 0.5 * (cov + cov.T)


def test_precond_matches_pinv_on_ill_conditioned() -> None:
    """HONEST positive control. For a single DENSE solve, np.linalg.pinv is ALREADY an
    eigenvalue-floored inverse (rcond relative to the largest SV), so the Hessian
    preconditioner is ALGEBRAICALLY the SAME Newton step — there is no convergence win to
    fake. What the preconditioner adds is (a) a gauge-explicit relative floor and (b) the
    condition-number sensor. This test asserts BOTH truths: the step matches pinv on a
    genuinely ill-conditioned (cond ~1e6) Hessian AND the sensor flags it."""
    cov = _ill_conditioned_softmax_cov([1e-6, 1e-4, 1e-2, 1.0])  # range cond = 1e6
    rng = np.random.default_rng(3)
    g = rng.standard_normal(5)
    g = g - g.mean()  # in the range (orthogonal to the gauge nullspace)
    taus, rcond = 1.0, 1e-10
    step_pre, cond = _newton_step_from_cov(cov, g, taus, precondition=True,
                                           rcond=rcond, eps_rel=1e-9, cond_gate=None)
    step_pin, _ = _newton_step_from_cov(cov, g, taus, precondition=False,
                                        rcond=rcond, eps_rel=1e-9, cond_gate=None)
    assert np.all(np.isfinite(step_pre))
    # relative match (steps are large under ill-conditioning; both are the SAME floored inverse)
    assert np.allclose(step_pre, step_pin, rtol=1e-7, atol=0.0), \
        "precond step must match the pinv step (dense solve is algebraically the same Newton step)"
    assert cond > 1e5, "the sensor must flag the constructed ill-conditioned Hessian"


def test_condition_number_sensor() -> None:
    # isotropic range (identity-like on the non-gauge subspace) => ~1
    m = np.full(5, 0.2)
    cov_iso = np.diag(m) - np.outer(m, m)  # rank-4, eigenvalues equal on the range
    c_iso = softmax_cov_condition_number(cov_iso)
    assert 0.99 <= c_iso <= 1.5
    # near-rank-1 (one class dominates) => large condition number
    s = np.zeros((10, 5))
    s[:, 2] = 1.0
    s[0, 0] = 1.0  # a single off pixel to avoid exact rank-1
    s = s / s.sum(axis=1, keepdims=True)
    mm = s.mean(axis=0)
    cov_deg = np.diag(mm) - (s.T @ s) / s.shape[0]
    assert softmax_cov_condition_number(cov_deg) > 10.0
    # exact rank-1 (all identical) => effective range empty beyond gauge => +inf
    s1 = np.zeros((10, 5))
    s1[:, 2] = 1.0
    mm1 = s1.mean(axis=0)
    cov_r1 = np.diag(mm1) - (s1.T @ s1) / s1.shape[0]
    assert softmax_cov_condition_number(cov_r1) == float("inf")


def test_zero_sum_and_determinism() -> None:
    phi, tgt = _well_conditioned()
    b1, _ = damped_newton_ot_offsets(phi, tgt, precondition=True)
    b2, _ = damped_newton_ot_offsets(phi, tgt, precondition=True)
    assert np.array_equal(b1, b2), "preconditioned solve must be deterministic"
    assert abs(float(b1.mean())) < 1e-9, "offset must stay zero-sum (gauge)"
