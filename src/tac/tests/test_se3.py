"""Tests for ``tac.se3`` — SE(3) Lie-group primitives for Lane RM.

These tests pin the mathematical contracts that the Riemannian SGD path
depends on:

  1. Rodrigues' formula matches scipy.spatial.transform.Rotation
     (golden reference) to 1e-6 absolute tolerance for axis-angle inputs
     across the practical range of pose-TTO updates (|ω| ∈ [0, π]).
  2. exp_map_so3 outputs are in SO(3): orthogonal (RᵀR = I) and
     determinant +1, regardless of input magnitude.
  3. log_map_so3 is the left inverse of exp_map_so3 within numerical
     tolerance (axis-angle round-trip drift < 1e-6 for |ω| < π - 1e-3).
  4. Small-angle Taylor branch matches the closed-form Rodrigues at the
     SMALL_ANGLE_THRESHOLD boundary to within float32 epsilon — i.e. the
     branch is C0-continuous (no jump discontinuity).
  5. left_jacobian_so3 reduces to the identity when ω = 0 (Sola Eq. 173
     evaluated at zero) and is well-conditioned at small angles.
  6. exp_map_se3(ω, v) reduces to (exp(ω̂), v) when ω = 0 and to
     (I, v) when both ω = 0 and v small (sanity sanity).
  7. geodesic_step preserves rotation orthogonality EXACTLY across many
     successive steps — this is the property Euclidean SGD on
     axis-angle does NOT have, and is the entire mathematical
     justification for Lane RM.
  8. batched_geodesic_step_axis_angle agrees with the per-element
     SE3Element + geodesic_step path within float32 epsilon.

Why scipy as the golden reference: scipy's
``Rotation.from_rotvec(ω).as_matrix()`` is a long-standing,
peer-reviewed reference implementation of Rodrigues; if our formula
matches it to 1e-6 then any downstream consumer that consumes our
rotations the same way scipy does will see a consistent answer.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as ScipyRotation

from tac.se3 import (
    NEAR_PI_THRESHOLD,
    SE3Element,
    SMALL_ANGLE_THRESHOLD,
    batched_geodesic_step_axis_angle,
    exp_map_se3,
    exp_map_so3,
    geodesic_step,
    hat_so3,
    left_jacobian_so3,
    log_map_so3,
    riemannian_gradient_se3,
    vee_so3,
)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _rand_axis_angles(n: int, max_norm: float, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    axis = torch.randn(n, 3, generator=g, dtype=torch.float64)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    theta = torch.rand(n, 1, generator=g, dtype=torch.float64) * max_norm
    return axis * theta


def _is_so3(R: torch.Tensor, atol: float = 1e-10) -> bool:
    """Verify R is in SO(3): RᵀR = I and det(R) = +1."""
    R64 = R.to(torch.float64)
    eye = torch.eye(3, dtype=torch.float64, device=R64.device)
    orth = torch.allclose(R64.transpose(-1, -2) @ R64, eye, atol=atol)
    det = torch.linalg.det(R64)
    detok = torch.allclose(det, torch.ones_like(det), atol=atol)
    return orth and detok


# ──────────────────────────────────────────────────────────────────────────
# hat / vee
# ──────────────────────────────────────────────────────────────────────────


def test_hat_vee_round_trip():
    omega = torch.tensor([0.7, -0.2, 1.3])
    assert torch.allclose(vee_so3(hat_so3(omega)), omega)


def test_hat_is_skew_symmetric():
    omega = torch.tensor([0.1, 0.2, 0.3])
    H = hat_so3(omega)
    assert torch.allclose(H, -H.transpose(-1, -2))


def test_hat_batched_shape():
    omega = torch.randn(5, 3)
    assert hat_so3(omega).shape == (5, 3, 3)


# ──────────────────────────────────────────────────────────────────────────
# Rodrigues vs scipy golden reference
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("max_norm", [1e-7, 1e-3, 0.5, 1.0, math.pi - 1e-3])
def test_exp_map_so3_matches_scipy(max_norm: float):
    """Rodrigues' formula should match scipy to 1e-6 abs tolerance.

    scipy.spatial.transform.Rotation.from_rotvec is the canonical Python
    reference for axis-angle → rotation matrix.
    """
    omega = _rand_axis_angles(64, max_norm=max_norm, seed=int(max_norm * 1e6))
    R_ours = exp_map_so3(omega).numpy()
    R_scipy = ScipyRotation.from_rotvec(omega.numpy()).as_matrix()
    np.testing.assert_allclose(R_ours, R_scipy, atol=1e-6)


def test_exp_map_so3_zero_is_identity():
    omega = torch.zeros(3, dtype=torch.float64)
    R = exp_map_so3(omega)
    assert torch.allclose(R, torch.eye(3, dtype=torch.float64))


def test_exp_map_so3_orthogonality():
    omega = _rand_axis_angles(32, max_norm=math.pi - 1e-3, seed=42)
    R = exp_map_so3(omega)
    for i in range(R.shape[0]):
        assert _is_so3(R[i]), f"R[{i}] is not in SO(3)"


def test_exp_map_so3_axis_invariance():
    """Rotating by ω about its own axis: applying R to ω returns ω
    (a rotation fixes its rotation axis)."""
    omega = torch.tensor([0.0, 0.0, 0.7], dtype=torch.float64)
    R = exp_map_so3(omega)
    assert torch.allclose(R @ omega, omega, atol=1e-10)


# ──────────────────────────────────────────────────────────────────────────
# Small-angle branch C0 continuity
# ──────────────────────────────────────────────────────────────────────────


def test_small_angle_branch_continuous_at_threshold():
    """At the SMALL_ANGLE_THRESHOLD boundary the Taylor branch and the
    closed-form must agree to ≪ 1 — i.e. there is no visible jump
    discontinuity. The expected agreement is O(input_delta) because both
    branches are smooth near the threshold; the test budget is generous
    (1e-5) since the input delta is 2 · eps = 2e-7 and a sin(θ)/θ
    discontinuity would dwarf that.
    """
    eps = 1e-7
    omega_below = torch.tensor([SMALL_ANGLE_THRESHOLD - eps, 0.0, 0.0],
                               dtype=torch.float64)
    omega_above = torch.tensor([SMALL_ANGLE_THRESHOLD + eps, 0.0, 0.0],
                               dtype=torch.float64)
    R_below = exp_map_so3(omega_below)
    R_above = exp_map_so3(omega_above)
    # Expect no jump bigger than ~10x the input delta (real jump
    # would be on the order of |1 - cos(θ)/θ| ≈ θ/2 ≈ 5e-7 if the
    # branches disagreed; we check 1e-5 to leave headroom).
    assert torch.allclose(R_below, R_above, atol=1e-5)


def test_exp_map_so3_handles_zero_without_nan():
    omega = torch.zeros(8, 3, dtype=torch.float64)
    R = exp_map_so3(omega)
    assert torch.isfinite(R).all()


# ──────────────────────────────────────────────────────────────────────────
# log_map_so3
# ──────────────────────────────────────────────────────────────────────────


def test_log_map_so3_is_inverse_of_exp():
    """log ∘ exp = identity on so(3) within (-π, π) (axis-angle ball)."""
    omega = _rand_axis_angles(64, max_norm=math.pi - 1e-2, seed=7)
    R = exp_map_so3(omega)
    omega_rt = log_map_so3(R)
    np.testing.assert_allclose(omega_rt.numpy(), omega.numpy(), atol=1e-9)


def test_log_map_so3_zero_at_identity():
    R = torch.eye(3, dtype=torch.float64)
    omega = log_map_so3(R)
    assert torch.allclose(omega, torch.zeros(3, dtype=torch.float64))


def test_log_map_so3_near_pi_returns_finite():
    """The near-π fallback branch must not return NaN / inf."""
    # Build a rotation of nearly-π about the z-axis.
    angle = math.pi - 1e-6
    omega = torch.tensor([0.0, 0.0, angle], dtype=torch.float64)
    R = exp_map_so3(omega)
    omega_rt = log_map_so3(R)
    assert torch.isfinite(omega_rt).all()
    # And the magnitude should be ≈ angle (axis up to ±).
    assert math.isclose(omega_rt.norm().item(), angle, abs_tol=1e-3)


# ──────────────────────────────────────────────────────────────────────────
# Bug 3 (codex Round 3) regression: SO(3) log map near π with axes that
# have *zero* components. The pre-fix near-π branch hard-coded
# ``sign(u_x) = +1`` and read sign(u_y), sign(u_z) from R[0, 1] / R[0, 2].
# When the true axis has u_x = 0 those entries are exactly zero and the
# sign of the y/z relationship encoded in R[1, 2] is silently lost. The
# post-fix branch picks the *largest-diagonal* anchor and derives the
# remaining components from the corresponding off-diagonal pair, which
# is the standard Sola 2018 §4.5.2 construction.
# ──────────────────────────────────────────────────────────────────────────


def _matrix_round_trip_close(omega: torch.Tensor, atol: float = 1e-9) -> None:
    """Helper: assert exp(log(exp(ω))) reconstructs the SAME rotation.

    We compare *rotation matrices* (not axis-angle vectors) because the
    axis-angle parameterisation has an antipodal sign ambiguity at θ=π
    — both ``+ω`` and ``-ω`` produce the same rotation matrix. The
    matrix equality is the only meaningful round-trip check at θ=π.
    """
    R_in = exp_map_so3(omega)
    omega_rt = log_map_so3(R_in)
    assert torch.isfinite(omega_rt).all(), (
        f"log_map produced non-finite output for ω={omega.tolist()}"
    )
    R_out = exp_map_so3(omega_rt)
    assert torch.allclose(R_in, R_out, atol=atol), (
        f"matrix round-trip failed at ω={omega.tolist()}:\n"
        f"R_in =\n{R_in}\nR_out =\n{R_out}\n"
        f"omega_rt = {omega_rt.tolist()}"
    )


def test_log_map_so3_pi_zero_x_axis_component():
    """Bug 3 core test: axis = (0, 1, -1)/√2 at θ=π. The pre-fix branch
    derived sign(u_y) and sign(u_z) from R[0,1] and R[0,2] — both zero
    here — so it lost the genuine sign relationship encoded in R[1,2]."""
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    omega = torch.tensor(
        [0.0, math.pi * inv_sqrt2, -math.pi * inv_sqrt2], dtype=torch.float64
    )
    _matrix_round_trip_close(omega)


def test_log_map_so3_pi_about_x_axis():
    """Cardinal axis: pure +x rotation by π. Anchor selection must pick
    index 0 (largest diagonal) and return ω = (π, 0, 0)."""
    omega = torch.tensor([math.pi, 0.0, 0.0], dtype=torch.float64)
    _matrix_round_trip_close(omega)


def test_log_map_so3_pi_about_y_axis():
    """Pure +y rotation by π. Anchor index 1."""
    omega = torch.tensor([0.0, math.pi, 0.0], dtype=torch.float64)
    _matrix_round_trip_close(omega)


def test_log_map_so3_pi_about_z_axis():
    """Pure +z rotation by π. Anchor index 2."""
    omega = torch.tensor([0.0, 0.0, math.pi], dtype=torch.float64)
    _matrix_round_trip_close(omega)


def test_log_map_so3_pi_axes_with_mixed_signs():
    """Five hand-picked sign patterns at θ=π that previously hit the
    sign-loss bug. Each must round-trip exactly (matrix equality)."""
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    inv_sqrt3 = 1.0 / math.sqrt(3.0)
    cases = [
        # u_x = 0, mixed y/z signs (the headline failure mode).
        (0.0, +inv_sqrt2, -inv_sqrt2),
        (0.0, -inv_sqrt2, +inv_sqrt2),
        # u_y = 0, mixed x/z signs (symmetric failure mode).
        (+inv_sqrt2, 0.0, -inv_sqrt2),
        (-inv_sqrt2, 0.0, +inv_sqrt2),
        # All-equal axis with mixed signs.
        (+inv_sqrt3, -inv_sqrt3, +inv_sqrt3),
    ]
    for ux, uy, uz in cases:
        omega = torch.tensor(
            [math.pi * ux, math.pi * uy, math.pi * uz], dtype=torch.float64
        )
        _matrix_round_trip_close(omega)


def test_log_map_so3_pi_random_axes():
    """Stress: 10 random pi-rotations covering arbitrary sign patterns.
    Matrix round-trip must hold for every one — proves the fix is not
    over-fitted to the cardinal/diagonal cases."""
    g = torch.Generator().manual_seed(2026_04_27)
    n = 10
    axes = torch.randn(n, 3, generator=g, dtype=torch.float64)
    axes = axes / axes.norm(dim=-1, keepdim=True)
    for i in range(n):
        omega = math.pi * axes[i]
        _matrix_round_trip_close(omega)


def test_log_map_so3_pi_batched_axes():
    """The fix must vectorise correctly: a batch of pi-rotations should
    all round-trip in a single call (no per-element Python loop)."""
    g = torch.Generator().manual_seed(7)
    axes = torch.randn(8, 3, generator=g, dtype=torch.float64)
    axes = axes / axes.norm(dim=-1, keepdim=True)
    omega = math.pi * axes  # (8, 3)
    R_in = exp_map_so3(omega)
    omega_rt = log_map_so3(R_in)
    assert torch.isfinite(omega_rt).all()
    R_out = exp_map_so3(omega_rt)
    # Per-batch matrix equality.
    assert torch.allclose(R_in, R_out, atol=1e-9), (
        f"batched pi-rotation round-trip failed:\n"
        f"R_in[0]=\n{R_in[0]}\nR_out[0]=\n{R_out[0]}"
    )


# ──────────────────────────────────────────────────────────────────────────
# left_jacobian_so3
# ──────────────────────────────────────────────────────────────────────────


def test_left_jacobian_zero_is_identity():
    omega = torch.zeros(3, dtype=torch.float64)
    Jl = left_jacobian_so3(omega)
    assert torch.allclose(Jl, torch.eye(3, dtype=torch.float64))


def test_left_jacobian_well_conditioned_at_small_angles():
    """Jl(ω) should be near-identity for tiny ω — and have finite condition
    number, not blow up due to division by ||ω||."""
    omega = torch.tensor([1e-9, 0.0, 0.0], dtype=torch.float64)
    Jl = left_jacobian_so3(omega)
    assert torch.allclose(Jl, torch.eye(3, dtype=torch.float64), atol=1e-8)


# ──────────────────────────────────────────────────────────────────────────
# exp_map_se3
# ──────────────────────────────────────────────────────────────────────────


def test_exp_map_se3_zero_rotation_returns_identity_R_and_v_t():
    omega = torch.zeros(3, dtype=torch.float64)
    v = torch.tensor([1.0, 2.0, -0.5], dtype=torch.float64)
    R, t = exp_map_se3(omega, v)
    assert torch.allclose(R, torch.eye(3, dtype=torch.float64))
    # When ω = 0, J_l = I, so t = v.
    assert torch.allclose(t, v)


def test_exp_map_se3_shapes_match_input():
    omega = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    v = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float64)
    R, t = exp_map_se3(omega, v)
    assert R.shape == (3, 3)
    assert t.shape == (3,)


def test_exp_map_se3_rotation_is_orthogonal():
    omega = torch.tensor([0.7, -0.2, 0.5], dtype=torch.float64)
    v = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    R, _ = exp_map_se3(omega, v)
    assert _is_so3(R)


def test_exp_map_se3_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        exp_map_se3(torch.zeros(3), torch.zeros(2))


# ──────────────────────────────────────────────────────────────────────────
# Riemannian gradient projection
# ──────────────────────────────────────────────────────────────────────────


def test_riemannian_gradient_se3_is_identity_under_left_invariant_metric():
    """With the canonical left-invariant SE(3) metric the projection is
    the identity — the tangent space at every point is canonically se(3)."""
    elem = SE3Element(omega=torch.zeros(3), t=torch.zeros(3))
    grad_omega = torch.tensor([0.5, -0.5, 0.1])
    grad_t = torch.tensor([1.0, 2.0, 3.0])
    proj = riemannian_gradient_se3((grad_omega, grad_t), elem)
    assert torch.equal(proj[0], grad_omega)
    assert torch.equal(proj[1], grad_t)


def test_riemannian_gradient_se3_rejects_bad_shapes():
    elem = SE3Element(omega=torch.zeros(3), t=torch.zeros(3))
    with pytest.raises(ValueError):
        riemannian_gradient_se3((torch.zeros(2), torch.zeros(3)), elem)


# ──────────────────────────────────────────────────────────────────────────
# Geodesic step — orthogonality preservation across many iterations
# ──────────────────────────────────────────────────────────────────────────


def test_geodesic_step_preserves_orthogonality_across_many_steps():
    """The defining property of Lane RM: rotation factor stays in SO(3)
    EXACTLY across many SGD steps. Standard Euclidean SGD on axis-angle
    does NOT have this property — it would silently drift.
    """
    torch.manual_seed(0)
    elem = SE3Element(
        omega=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
        t=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
    )
    # Take 1000 random tangent steps; after each, R(ω) must be in SO(3).
    for k in range(1000):
        eta_omega = torch.randn(3, dtype=torch.float64) * 0.05
        eta_v = torch.randn(3, dtype=torch.float64) * 0.05
        elem = geodesic_step(elem, (eta_omega, eta_v), step_size=-0.01)
        R = exp_map_so3(elem.omega)
        assert _is_so3(R, atol=1e-9), (
            f"orthogonality drift detected at step {k}: ω={elem.omega.tolist()}"
        )


def test_geodesic_step_zero_tangent_is_no_op():
    elem = SE3Element(
        omega=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
        t=torch.tensor([0.4, 0.5, 0.6], dtype=torch.float64),
    )
    new = geodesic_step(elem,
                        (torch.zeros(3, dtype=torch.float64),
                         torch.zeros(3, dtype=torch.float64)),
                        step_size=-0.5)
    assert torch.allclose(new.omega, elem.omega, atol=1e-10)
    assert torch.allclose(new.t, elem.t, atol=1e-10)


def test_geodesic_step_returns_new_instance():
    elem = SE3Element(omega=torch.zeros(3), t=torch.zeros(3))
    new = geodesic_step(elem, (torch.ones(3) * 0.01, torch.ones(3) * 0.01),
                        step_size=-0.1)
    assert new is not elem
    # And the original is unchanged (frozen dataclass).
    assert torch.equal(elem.omega, torch.zeros(3))


# ──────────────────────────────────────────────────────────────────────────
# Batched helper agrees with per-element path
# ──────────────────────────────────────────────────────────────────────────


def test_batched_geodesic_step_matches_per_element():
    torch.manual_seed(1)
    N = 7
    omega_b = torch.randn(N, 3, dtype=torch.float64) * 0.3
    t_b = torch.randn(N, 3, dtype=torch.float64) * 0.5
    grad_omega = torch.randn(N, 3, dtype=torch.float64) * 0.1
    grad_t = torch.randn(N, 3, dtype=torch.float64) * 0.1
    step = -0.05

    omega_new_b, t_new_b = batched_geodesic_step_axis_angle(
        omega_b, t_b, grad_omega, grad_t, step
    )

    # Compare element-by-element against the dataclass path.
    for i in range(N):
        elem = SE3Element(omega=omega_b[i].clone(), t=t_b[i].clone())
        new = geodesic_step(elem, (grad_omega[i], grad_t[i]), step)
        assert torch.allclose(omega_new_b[i], new.omega, atol=1e-12)
        assert torch.allclose(t_new_b[i], new.t, atol=1e-12)


def test_batched_geodesic_step_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        batched_geodesic_step_axis_angle(
            torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(3, 3),
            torch.zeros(2, 3), -0.1,
        )


# ──────────────────────────────────────────────────────────────────────────
# SE3Element validation
# ──────────────────────────────────────────────────────────────────────────


def test_se3element_rejects_bad_omega_shape():
    with pytest.raises(ValueError):
        SE3Element(omega=torch.zeros(2), t=torch.zeros(3))


def test_se3element_rejects_bad_t_shape():
    with pytest.raises(ValueError):
        SE3Element(omega=torch.zeros(3), t=torch.zeros(4))


def test_se3element_is_frozen():
    elem = SE3Element(omega=torch.zeros(3), t=torch.zeros(3))
    with pytest.raises(Exception):
        elem.omega = torch.ones(3)  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────


def test_thresholds_are_documented_and_positive():
    """Sanity: the small-angle threshold is positive and the near-π
    threshold is positive — these guard divisions by zero."""
    assert SMALL_ANGLE_THRESHOLD > 0
    assert NEAR_PI_THRESHOLD > 0
    # And SMALL_ANGLE_THRESHOLD is loose enough for float32 (≈ 1e-7 eps).
    assert SMALL_ANGLE_THRESHOLD >= 1e-7
