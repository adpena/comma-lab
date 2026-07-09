# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the #288 selectable head-offset SOLVER dispatcher + the damped-Newton
semi-discrete OT offset (``solve_head_offsets`` / ``damped_newton_ot_offsets``).

These pin the REAL properties the #288 wire-in depends on: the OT solve actually matches the target
cell masses (mass-matching property, on real synthetic phi), the offsets are zero-sum, ``ot_newton``
NEVER silently falls back to the Menon prior (the no-fake contract), and the byte-free out_sdf.bias
fold is the exact ``argmax(phi+b)`` identity."""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.laguerre_logit_offset import (
    HEAD_OFFSET_SOLVERS,
    LaguerreLogitOffsetError,
    apply_offset_to_sdf_bias,
    damped_newton_ot_offsets,
    menon_logit_adjustment_offsets,
    power_diagram_argmax,
    soft_cell_masses,
    solve_head_offsets,
)


def _synthetic_phi(seed: int = 0, n: int = 4000, k: int = 5) -> np.ndarray:
    """A REAL (not degenerate) logit field with an imbalanced argmax distribution."""
    rng = np.random.default_rng(seed)
    phi = rng.standard_normal((n, k))
    phi[:, 0] += 1.5  # make class 0 dominate the un-offset argmax (imbalance to correct)
    return phi


# ---- OT solver: the mass-matching PROPERTY (the whole point) ----------------
def test_ot_solve_matches_target_masses():
    phi = _synthetic_phi()
    target = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    b, info = damped_newton_ot_offsets(phi, target, tau=1.0)
    m = soft_cell_masses(phi, b, tau=1.0)
    assert np.allclose(m, target, atol=1e-6), (m, target)
    assert info["converged"] == 1.0
    assert info["max_mass_err"] <= 1e-8


def test_ot_solve_matches_imbalanced_target_masses():
    phi = _synthetic_phi(seed=3)
    target = np.array([0.05, 0.4, 0.05, 0.4, 0.1])  # deliberately non-uniform
    b, info = damped_newton_ot_offsets(phi, target, tau=0.7)
    m = soft_cell_masses(phi, b, tau=0.7)
    assert np.allclose(m, target, atol=1e-5), (m, target)


def test_ot_offsets_zero_sum():
    phi = _synthetic_phi(seed=1)
    b, _ = damped_newton_ot_offsets(phi, np.full(5, 0.2), tau=1.0)
    assert abs(float(b.sum())) < 1e-9


def test_ot_solve_renormalizes_unnormalized_masses():
    """Target masses need not sum to 1 (raw class COUNTS are accepted)."""
    phi = _synthetic_phi(seed=2)
    counts = np.array([100.0, 5.0, 200.0, 12.0, 90.0])  # e.g. GT class pixel counts
    b, _ = damped_newton_ot_offsets(phi, counts, tau=1.0)
    m = soft_cell_masses(phi, b, tau=1.0)
    assert np.allclose(m, counts / counts.sum(), atol=1e-5)


def test_ot_rejects_bad_masses():
    phi = _synthetic_phi()
    with pytest.raises(LaguerreLogitOffsetError):
        damped_newton_ot_offsets(phi, np.array([-1.0, 1.0, 1.0, 1.0, 1.0]), tau=1.0)
    with pytest.raises(LaguerreLogitOffsetError):
        damped_newton_ot_offsets(phi, np.zeros(5), tau=1.0)
    with pytest.raises(LaguerreLogitOffsetError):
        damped_newton_ot_offsets(phi, np.array([0.5, 0.5]), tau=1.0)  # K mismatch


# ---- the selectable dispatcher: solve_head_offsets --------------------------
def test_dispatcher_menon_equals_menon_helper():
    priors = np.array([0.232, 0.0059, 0.495, 0.0124, 0.254])
    b_disp, info = solve_head_offsets("menon", priors=priors, tau=1.3)
    b_ref = menon_logit_adjustment_offsets(priors, tau=1.3)
    assert np.array_equal(b_disp, b_ref)
    assert info["solver"] == 0.0


def test_dispatcher_menon_accepts_target_masses_alias():
    counts = np.array([100.0, 3.0, 200.0, 6.0, 90.0])
    b_alias, _ = solve_head_offsets("menon", target_masses=counts)
    b_ref = menon_logit_adjustment_offsets(counts, tau=1.0)
    assert np.array_equal(b_alias, b_ref)


def test_dispatcher_menon_requires_some_priors():
    with pytest.raises(LaguerreLogitOffsetError):
        solve_head_offsets("menon")


def test_dispatcher_ot_newton_runs_real_solver():
    phi = _synthetic_phi(seed=7)
    counts = np.array([100.0, 5.0, 200.0, 12.0, 90.0])
    b, info = solve_head_offsets("ot_newton", phi=phi, target_masses=counts, tau=1.0)
    assert info["solver"] == 1.0
    m = soft_cell_masses(phi, b, tau=1.0)
    assert np.allclose(m, counts / counts.sum(), atol=1e-5)


def test_dispatcher_ot_newton_requires_phi_no_fake():
    """ot_newton with NO phi must RAISE — never silently degenerate to the Menon prior (that would be
    a fake 'ot_newton' ignoring the geometry it claims to use)."""
    counts = np.array([100.0, 5.0, 200.0, 12.0, 90.0])
    with pytest.raises(LaguerreLogitOffsetError):
        solve_head_offsets("ot_newton", priors=counts, target_masses=counts)  # phi missing


def test_dispatcher_ot_newton_requires_target_masses_no_fake():
    phi = _synthetic_phi()
    with pytest.raises(LaguerreLogitOffsetError):
        solve_head_offsets("ot_newton", phi=phi)  # target_masses missing


def test_dispatcher_unknown_mode_raises():
    with pytest.raises(LaguerreLogitOffsetError):
        solve_head_offsets("laguerre_magic", priors=np.full(5, 0.2))
    assert set(HEAD_OFFSET_SOLVERS) == {"menon", "ot_newton"}


def test_ot_differs_from_menon_on_real_geometry():
    """The whole #288 premise: OT uses the phi geometry, so its offset is NOT the priors-only Menon
    offset (they would coincide only if the witness argmax were already perfectly balanced)."""
    phi = _synthetic_phi(seed=11)
    counts = np.array([100.0, 5.0, 200.0, 12.0, 90.0])
    b_ot, _ = solve_head_offsets("ot_newton", phi=phi, target_masses=counts, tau=1.0)
    b_menon, _ = solve_head_offsets("menon", priors=counts, tau=1.0)
    assert not np.allclose(b_ot, b_menon, atol=1e-3)


# ---- the byte-free fold identity (the decode-time application) --------------
def test_ot_offset_fold_is_byte_free_argmax_identity():
    """Folding b* into out_sdf.bias == adding b* to phi at argmax (the byte-free decode identity)."""
    rng = np.random.default_rng(5)
    phi = rng.standard_normal((50, 5))
    counts = np.array([100.0, 5.0, 200.0, 12.0, 90.0])
    b, _ = solve_head_offsets("ot_newton", phi=phi, target_masses=counts, tau=1.0)
    params = {"out_sdf.bias": np.zeros(5, np.float32), "out_sdf.weight": np.eye(5, dtype=np.float32)}
    folded = apply_offset_to_sdf_bias(params, b)
    # argmax(phi + b) via power_diagram == argmax(phi + folded_bias) since weight is identity
    lab_offset = power_diagram_argmax(phi, b)
    lab_folded = np.argmax(phi @ folded["out_sdf.weight"].T + folded["out_sdf.bias"], axis=-1)
    assert np.array_equal(lab_offset, lab_folded)
    # and the fold did NOT mutate the input params (copy semantics)
    assert np.array_equal(params["out_sdf.bias"], np.zeros(5, np.float32))


def test_ot_symmetric_phi_uniform_target_gives_near_zero_offset():
    """Degeneracy: a symmetric phi already at uniform masses needs ~no offset to hit uniform target."""
    rng = np.random.default_rng(9)
    phi = rng.standard_normal((3000, 5))  # no class boosted => argmax already ~uniform
    b, info = solve_head_offsets("ot_newton", phi=phi, target_masses=np.full(5, 0.2), tau=1.0)
    assert info["converged"] == 1.0
    assert float(np.max(np.abs(b))) < 0.5  # small correction only
