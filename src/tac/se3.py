"""SE(3) Lie group operations for Riemannian pose optimization (Lane RM).

This module implements the matrix Lie group SE(3) = SO(3) ⋉ ℝ³ with the
exponential and logarithm maps used by the Riemannian SGD optimizer in
``src/tac/riemannian_pose_optimizer.py``.

References
----------
* Absil, Mahony, Sepulchre, *Optimization Algorithms on Matrix Manifolds*
  (Princeton University Press, 2008), Chapter 3 (manifolds), Chapter 4
  (line-search algorithms on Riemannian manifolds).
* Boumal, *An Introduction to Optimization on Smooth Manifolds*
  (Cambridge University Press, 2023), Chapters 3 (manifold structure) and
  10 (Riemannian SGD).
* Bonnabel, *Stochastic gradient descent on Riemannian manifolds*, IEEE
  TAC 58(9), 2013 — convergence rate matches Euclidean SGD on smooth
  manifolds while preserving the manifold constraint.
* Sola, Deray, Atchuthan, *A micro Lie theory for state estimation in
  robotics*, arXiv:1812.01537, 2018 — closed-form SE(3) exp / log,
  numerical stability of small-angle expansions, chosen as the canonical
  reference for the formulae used below.

Conventions
-----------
* A pose tangent vector ``ξ ∈ se(3)`` is split as ``(ω, v)`` where ``ω ∈ ℝ³``
  is the rotation tangent (axis-angle) and ``v ∈ ℝ³`` is the translation
  tangent. We follow the *Sola et al.* ordering (rotation first, then
  translation), which matches how the comma.ai PoseNet head emits its
  6-vector output (the first three dims are angular, the last three
  translational, per ``upstream/openpilot/selfdrive/modeld/models``).
* SO(3) is parameterised by the axis-angle ``ω`` so that an element
  ``R ∈ SO(3)`` is represented by ``ω = log_map_so3(R) ∈ ℝ³``.
* The small-angle threshold ``1e-6`` (radians) is motivated by the fact
  that ``sin(|ω|)/|ω| = 1 - |ω|²/6 + O(|ω|⁴)`` so the relative error of the
  closed-form Rodrigues coefficient with float32 precision crosses the
  machine-epsilon (≈ 1.19e-7) threshold around ``|ω| ≈ 1e-6``. Below this
  we evaluate the Taylor series instead of dividing by ``|ω|``, and above
  it the closed-form is more accurate than the Taylor truncation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

# ──────────────────────────────────────────────────────────────────────────
# Numerical-stability constants
# ──────────────────────────────────────────────────────────────────────────

# Below this rotation angle (radians) the closed-form Rodrigues coefficients
# (sin|ω|/|ω| and (1-cos|ω|)/|ω|²) lose precision in float32 because the
# numerator and denominator both go to zero. We switch to the truncated
# Taylor expansions which remain accurate to O(|ω|⁴) ≪ machine eps.
SMALL_ANGLE_THRESHOLD: float = 1.0e-6

# Used to guard the SO(3) log map at θ ≈ π where the closed-form
# axis = (R - Rᵀ) / (2 sinθ) becomes singular.
NEAR_PI_THRESHOLD: float = 1.0e-3


# ──────────────────────────────────────────────────────────────────────────
# SE(3) element dataclass
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SE3Element:
    """A single SE(3) pose stored in tangent-space coordinates.

    Attributes
    ----------
    omega : torch.Tensor of shape (3,)
        Axis-angle rotation (i.e. ``log_map_so3(R)``). The Lie algebra
        coordinate of the rotation component.
    t : torch.Tensor of shape (3,)
        Translation vector in ℝ³. Note that this is the *Cartesian*
        translation (the "t" in the matrix [R t; 0 1]); it is NOT the
        "v" of the SE(3) Lie algebra coordinate. We chose this convention
        because the renderer's pose head produces a Cartesian translation,
        not the tangent-space ``v``. Conversion between them is handled by
        ``exp_map_se3`` / ``log_map_se3`` when needed.

    Notes
    -----
    The dataclass is frozen so that an ``SE3Element`` can be safely used as
    an optimizer-state key and so that the geodesic step
    (``geodesic_step``) is forced to return a new instance rather than
    silently mutating an existing one.
    """

    omega: torch.Tensor
    t: torch.Tensor

    def __post_init__(self) -> None:
        # Use object.__setattr__ because the dataclass is frozen.
        if self.omega.shape != (3,):
            raise ValueError(
                f"omega must have shape (3,), got {tuple(self.omega.shape)}"
            )
        if self.t.shape != (3,):
            raise ValueError(
                f"t must have shape (3,), got {tuple(self.t.shape)}"
            )


# ──────────────────────────────────────────────────────────────────────────
# Hat / vee operators (so(3) ↔ skew-symmetric 3×3 matrices)
# ──────────────────────────────────────────────────────────────────────────


def hat_so3(omega: torch.Tensor) -> torch.Tensor:
    """Map ω ∈ ℝ³ to the corresponding skew-symmetric 3×3 matrix ω̂.

    Sola et al. (2018), Eq. (24)::

        ω̂ = [[  0  , -ω_z,  ω_y],
             [ ω_z,   0  , -ω_x],
             [-ω_y,  ω_x,   0  ]]
    """
    if omega.shape[-1] != 3:
        raise ValueError(f"hat_so3 expects last dim 3, got {tuple(omega.shape)}")
    x, y, z = omega[..., 0], omega[..., 1], omega[..., 2]
    zero = torch.zeros_like(x)
    row0 = torch.stack([zero, -z, y], dim=-1)
    row1 = torch.stack([z, zero, -x], dim=-1)
    row2 = torch.stack([-y, x, zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def vee_so3(omega_hat: torch.Tensor) -> torch.Tensor:
    """Inverse of ``hat_so3``: extract ω ∈ ℝ³ from a skew-symmetric matrix.

    Sola et al. (2018), Eq. (25). Antisymmetry is *assumed* (not enforced) —
    this matches the convention in ``log_map_so3`` where the input is
    ``(R - Rᵀ)`` which is antisymmetric by construction.
    """
    if omega_hat.shape[-2:] != (3, 3):
        raise ValueError(
            f"vee_so3 expects last two dims (3, 3), got {tuple(omega_hat.shape)}"
        )
    return torch.stack(
        [
            omega_hat[..., 2, 1],
            omega_hat[..., 0, 2],
            omega_hat[..., 1, 0],
        ],
        dim=-1,
    )


# ──────────────────────────────────────────────────────────────────────────
# SO(3) exponential map (Rodrigues' rotation formula)
# ──────────────────────────────────────────────────────────────────────────


def exp_map_so3(omega: torch.Tensor) -> torch.Tensor:
    """Rodrigues' rotation formula: so(3) → SO(3).

    Sola et al. (2018), Eq. (143); Absil-Mahony-Sepulchre §3.5::

        exp(ω̂) = I + (sin θ / θ) ω̂ + ((1 - cos θ) / θ²) ω̂²
        where θ = ‖ω‖.

    For θ < SMALL_ANGLE_THRESHOLD we substitute the Taylor expansions
    sin θ / θ ≈ 1 - θ²/6 and (1 - cos θ) / θ² ≈ 0.5 - θ²/24, which is
    accurate to O(θ⁴) and avoids the 0/0 numerical issue. Above this
    threshold we use the closed form, which is more accurate than the
    truncated Taylor series.

    Parameters
    ----------
    omega : torch.Tensor of shape (..., 3)
        Axis-angle vector(s).

    Returns
    -------
    torch.Tensor of shape (..., 3, 3)
        Corresponding rotation matrix / matrices.
    """
    if omega.shape[-1] != 3:
        raise ValueError(f"exp_map_so3 expects last dim 3, got {tuple(omega.shape)}")

    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)  # (..., 1)
    theta_sq = theta * theta

    # Closed-form coefficients, valid for θ above the small-angle threshold.
    # We compute them unconditionally and then ``torch.where`` against the
    # Taylor expansion so the computation stays differentiable everywhere.
    safe_theta = theta.clamp(min=SMALL_ANGLE_THRESHOLD)
    a_closed = torch.sin(safe_theta) / safe_theta                 # sin θ / θ
    b_closed = (1.0 - torch.cos(safe_theta)) / (safe_theta * safe_theta)  # (1 - cos θ) / θ²

    # Truncated Taylor coefficients (accurate to O(θ⁴)).
    a_taylor = 1.0 - theta_sq / 6.0
    b_taylor = 0.5 - theta_sq / 24.0

    small = theta < SMALL_ANGLE_THRESHOLD  # (..., 1) bool
    a = torch.where(small, a_taylor, a_closed)  # (..., 1)
    b = torch.where(small, b_taylor, b_closed)  # (..., 1)

    omega_hat = hat_so3(omega)  # (..., 3, 3)
    omega_hat_sq = omega_hat @ omega_hat  # (..., 3, 3)

    # Broadcast a, b from (..., 1) to (..., 1, 1) so they multiply matrices.
    a_mat = a.unsqueeze(-1)
    b_mat = b.unsqueeze(-1)

    eye = torch.eye(3, dtype=omega.dtype, device=omega.device).expand_as(omega_hat)
    return eye + a_mat * omega_hat + b_mat * omega_hat_sq


# ──────────────────────────────────────────────────────────────────────────
# SO(3) logarithm map (inverse of Rodrigues)
# ──────────────────────────────────────────────────────────────────────────


def log_map_so3(R: torch.Tensor) -> torch.Tensor:
    """SO(3) → so(3) (axis-angle), inverse of ``exp_map_so3``.

    Sola et al. (2018), Eq. (146). For a generic rotation::

        θ   = arccos((tr(R) - 1) / 2)
        ω̂   = θ / (2 sin θ) · (R - Rᵀ)
        ω   = vee(ω̂)

    Two singular cases are handled explicitly:

    * θ ≈ 0  (R ≈ I): ``θ / (2 sin θ) → 1/2`` (Taylor) so we just take
      ``ω = 0.5 · vee(R - Rᵀ)``.
    * θ ≈ π: 2 sin θ → 0 so the closed form is unstable. We extract the
      axis from the diagonal of ``R + I = 2 u uᵀ`` (Sola Eq. 147) and pick
      its sign from the off-diagonal entries.

    Parameters
    ----------
    R : torch.Tensor of shape (..., 3, 3)
        Rotation matrix / matrices. Must be in SO(3) (orthogonal, det +1)
        within numerical tolerance.

    Returns
    -------
    torch.Tensor of shape (..., 3)
        Axis-angle vector(s).
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"log_map_so3 expects (..., 3, 3), got {tuple(R.shape)}")

    # tr(R) is the sum of the diagonal elements.
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]  # (...,)
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)  # numerical safety
    theta = torch.acos(cos_theta)  # (...,) in [0, π]

    # Closed-form axis-from-skew (works away from θ=π).
    sin_theta = torch.sin(theta)  # (...,)
    safe_sin = sin_theta.clamp(min=SMALL_ANGLE_THRESHOLD)
    skew = R - R.transpose(-1, -2)  # (..., 3, 3)
    omega_closed = (theta / (2.0 * safe_sin)).unsqueeze(-1) * vee_so3(skew)  # (..., 3)

    # Small-angle Taylor: θ / (2 sin θ) → 1/2 + θ²/12 + ...
    # We use the leading 1/2 term — the error is O(θ²) which is < eps for
    # θ < SMALL_ANGLE_THRESHOLD.
    omega_small = 0.5 * vee_so3(skew)  # (..., 3)

    # Near-π fallback: derive the axis from R + I = 2 u uᵀ (Sola Eq. 147).
    #
    # Bug 3 fix (codex Round 3): the previous implementation hard-coded
    # ``sign(u_x) = +1`` and read ``sign(u_y) = sign(R[0,1])``,
    # ``sign(u_z) = sign(R[0,2])``. When the *true* axis has ``u_x = 0``
    # (e.g. axis = (0, 1, -1)/√2) the entries ``R[0,1]`` and ``R[0,2]``
    # are exactly ``2 · u_x · u_y = 0`` and ``2 · u_x · u_z = 0``
    # respectively, so the sign extraction defaults to +1 for both
    # components — and the genuine sign relationship between ``u_y``
    # and ``u_z`` (encoded in ``R[1,2] = 2 · u_y · u_z``) is silently
    # discarded. The reconstructed rotation is then wrong by a sign flip
    # on one component.
    #
    # The standard fix (Sola 2018 §4.5.2; Shoemake 1985 quaternion ↔
    # rotation matrix; Eberly *Game Engine Geometry* §6.5) is to extract
    # the axis component with the largest squared magnitude FIRST (call
    # it index ``i``), use the positive square root for that one, and
    # derive the remaining components from the corresponding off-diagonal
    # pair containing ``i``: ``u[j] = R[i, j] / (2 · u[i])`` for j ≠ i.
    # Picking the largest entry guarantees ``u[i]`` is bounded away from
    # zero so the division is numerically safe.
    rp = R + torch.eye(3, dtype=R.dtype, device=R.device).expand_as(R)
    # Squared axis components from the diagonal of (R + I) / 2 = u uᵀ.
    u_sq = torch.stack(
        [rp[..., 0, 0] / 2.0, rp[..., 1, 1] / 2.0, rp[..., 2, 2] / 2.0],
        dim=-1,
    ).clamp(min=0.0)  # (..., 3); negatives are pure floating-point noise.
    # Index of the largest squared component (per element of the batch).
    i_max = u_sq.argmax(dim=-1)  # (...,)
    # Positive sqrt of the largest squared component — the "anchor".
    u_max = torch.sqrt(u_sq.gather(-1, i_max.unsqueeze(-1))).squeeze(-1)
    # The two non-anchor components: u[j] = (R+I)[i_max, j] / (2 · u[i_max])
    # for j ≠ i_max.  Note that for j ≠ i_max we have
    #     (R+I)[i_max, j] = R[i_max, j] = 2 · u[i_max] · u[j]
    # so the division is exact. We use rp (= R+I) below for clarity. The
    # diagonal entry rp[i_max, i_max] is NOT used to derive u[j=i_max] —
    # that slot is filled by the (positive) sqrt anchor u_max instead, so
    # the resulting axis exactly satisfies u_axis[i_max] = u_max ≥ 0.
    expand_idx = i_max.unsqueeze(-1).unsqueeze(-1).expand(*i_max.shape, 1, 3)
    rp_anchor_row = rp.gather(-2, expand_idx).squeeze(-2)  # (..., 3)
    # Use 2*u_max with a small floor to avoid div-by-zero on the
    # vanishingly-rare case θ = π exactly with two equal anchors.
    safe_anchor = u_max.clamp(min=1e-12).unsqueeze(-1)
    u_other = rp_anchor_row / (2.0 * safe_anchor)  # (..., 3) — valid for j ≠ i_max
    # Compose the axis: place +u_max (the positive sqrt) at index i_max
    # and u_other at every other index. ``scatter_`` handles arbitrary
    # ``i_max`` per batch element.
    u_axis = u_other.clone()
    u_axis.scatter_(-1, i_max.unsqueeze(-1), u_max.unsqueeze(-1))
    omega_pi = u_axis * theta.unsqueeze(-1)

    # Compose the three branches.
    small = (theta < SMALL_ANGLE_THRESHOLD).unsqueeze(-1)  # (..., 1)
    near_pi = (torch.pi - theta < NEAR_PI_THRESHOLD).unsqueeze(-1)
    omega = torch.where(small, omega_small, omega_closed)
    omega = torch.where(near_pi, omega_pi, omega)
    return omega


# ──────────────────────────────────────────────────────────────────────────
# SE(3) exponential map
# ──────────────────────────────────────────────────────────────────────────


def left_jacobian_so3(omega: torch.Tensor) -> torch.Tensor:
    """Left Jacobian of SO(3) (Sola Eq. 173)::

        J_l(ω) = I + ((1 - cos θ) / θ²) ω̂ + ((θ - sin θ) / θ³) ω̂²

    This is the Jacobian that relates the SE(3) tangent translation ``v``
    to the Cartesian translation ``t`` via ``t = J_l(ω) · v``. Below the
    small-angle threshold we use Taylor expansions
    (1-cosθ)/θ² ≈ 1/2 - θ²/24 and (θ-sinθ)/θ³ ≈ 1/6 - θ²/120.
    """
    if omega.shape[-1] != 3:
        raise ValueError(
            f"left_jacobian_so3 expects last dim 3, got {tuple(omega.shape)}"
        )

    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)  # (..., 1)
    theta_sq = theta * theta
    safe_theta = theta.clamp(min=SMALL_ANGLE_THRESHOLD)

    a_closed = (1.0 - torch.cos(safe_theta)) / (safe_theta * safe_theta)  # (..., 1)
    b_closed = (safe_theta - torch.sin(safe_theta)) / (safe_theta ** 3)   # (..., 1)
    a_taylor = 0.5 - theta_sq / 24.0
    b_taylor = (1.0 / 6.0) - theta_sq / 120.0

    small = theta < SMALL_ANGLE_THRESHOLD
    a = torch.where(small, a_taylor, a_closed).unsqueeze(-1)  # (..., 1, 1)
    b = torch.where(small, b_taylor, b_closed).unsqueeze(-1)

    omega_hat = hat_so3(omega)  # (..., 3, 3)
    omega_hat_sq = omega_hat @ omega_hat  # (..., 3, 3)
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device).expand_as(omega_hat)
    return eye + a * omega_hat + b * omega_hat_sq


def exp_map_se3(omega: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Closed-form SE(3) exponential map (Sola Eq. 172)::

        exp([ω̂  v;  0  0]) = [exp(ω̂)   J_l(ω) v ;
                              0           1     ]

    The function returns the (R, t) pair so callers do not have to slice
    a 4×4 matrix.

    Parameters
    ----------
    omega : torch.Tensor of shape (..., 3)
        Rotation tangent (axis-angle).
    v : torch.Tensor of shape (..., 3)
        Translation tangent (so(3) coordinate, NOT Cartesian translation).

    Returns
    -------
    R : torch.Tensor of shape (..., 3, 3)
    t : torch.Tensor of shape (..., 3)
        Cartesian translation = J_l(ω) · v.
    """
    if omega.shape != v.shape:
        raise ValueError(
            f"exp_map_se3 expects matching shapes, got omega={tuple(omega.shape)} "
            f"v={tuple(v.shape)}"
        )
    R = exp_map_so3(omega)
    Jl = left_jacobian_so3(omega)
    t = (Jl @ v.unsqueeze(-1)).squeeze(-1)
    return R, t


# ──────────────────────────────────────────────────────────────────────────
# Riemannian gradient projection
# ──────────────────────────────────────────────────────────────────────────


def riemannian_gradient_se3(
    euclidean_grad: Tuple[torch.Tensor, torch.Tensor],
    current: SE3Element,  # noqa: ARG001 — reserved for non-trivial metrics
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project a Euclidean gradient onto the tangent space of SE(3).

    For SE(3) with the *left-invariant* metric (the canonical choice in
    Absil-Mahony-Sepulchre §3.6.1 and Boumal §3.7), the tangent space at
    every point is canonically isomorphic to se(3), so the projection is
    the identity: the Euclidean gradient on the parameter
    ``(omega, t) ∈ ℝ⁶`` is already in the tangent space.

    The current point is accepted as an argument because the abstraction
    needs a hook for non-trivial metrics (e.g. weighted SE(3) metrics or
    the right-invariant metric, which would multiply the gradient by an
    Adjoint matrix); under the canonical left-invariant metric the
    function is the identity.

    Parameters
    ----------
    euclidean_grad : (∂L/∂ω, ∂L/∂t) tensors of shape (3,) each.
    current : current SE3Element (unused — see above).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Riemannian gradient in se(3) coordinates.
    """
    grad_omega, grad_t = euclidean_grad
    if grad_omega.shape != (3,) or grad_t.shape != (3,):
        raise ValueError(
            "riemannian_gradient_se3 expects shape (3,) for each component"
        )
    return grad_omega, grad_t


# ──────────────────────────────────────────────────────────────────────────
# Geodesic step
# ──────────────────────────────────────────────────────────────────────────


def geodesic_step(
    current: SE3Element,
    tangent: Tuple[torch.Tensor, torch.Tensor],
    step_size: float,
) -> SE3Element:
    """Take a geodesic step on SE(3): retraction via the exponential map.

    Boumal §3.6 (retractions): ``Retr_x(η) = exp_x(η)`` is the canonical
    second-order retraction on a Lie group. For SE(3) this is::

        new_R = current_R · exp_so3(step · η_ω)
        new_t = current_t + step · η_v       (Cartesian translation; the
                                              tangent's translational
                                              component IS the Cartesian
                                              displacement when the
                                              left-invariant metric is
                                              chosen on ℝ³).

    The rotation update is the *exponential map composition* on SO(3); the
    rotation matrix is recomposed via ``R_new = R · exp(η̂)`` so the result
    is exactly orthogonal regardless of step size — this is the key
    property that Euclidean SGD on the (axis-angle, t) parameterisation
    fails to provide (it accumulates orthogonality drift on each step).

    Parameters
    ----------
    current : SE3Element
    tangent : tuple of (η_ω, η_v) tensors of shape (3,)
        The Riemannian gradient (after projection); see
        ``riemannian_gradient_se3``.
    step_size : float
        Geodesic step length (negative for descent).

    Returns
    -------
    SE3Element
        The new point on SE(3).
    """
    eta_omega, eta_v = tangent

    # Map the current axis-angle to the rotation matrix, compose with the
    # exponential of the geodesic increment, then read back the axis-angle.
    R_current = exp_map_so3(current.omega)
    delta_R = exp_map_so3(step_size * eta_omega)
    R_new = R_current @ delta_R
    omega_new = log_map_so3(R_new)

    # Translation is updated by a straight Euclidean step in ℝ³ — this is
    # the geodesic on (ℝ³, standard metric), which is the canonical metric
    # on the translational factor of SE(3) in the left-invariant
    # decomposition.
    t_new = current.t + step_size * eta_v

    return SE3Element(omega=omega_new, t=t_new)


# ──────────────────────────────────────────────────────────────────────────
# Batched helpers — used by the optimiser hot path
# ──────────────────────────────────────────────────────────────────────────


def batched_geodesic_step_axis_angle(
    omega_batch: torch.Tensor,
    t_batch: torch.Tensor,
    grad_omega: torch.Tensor,
    grad_t: torch.Tensor,
    step_size: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorised geodesic step for a batch of SE(3) elements.

    All inputs are batched along leading dimensions; the last dim is 3
    (axis-angle / translation). This avoids the Python-level loop inside
    the optimizer for performance.

    Returns updated ``(omega, t)`` tensors with the same shape as the
    inputs.
    """
    if omega_batch.shape != grad_omega.shape:
        raise ValueError(
            f"omega_batch shape {tuple(omega_batch.shape)} != "
            f"grad_omega shape {tuple(grad_omega.shape)}"
        )
    if t_batch.shape != grad_t.shape:
        raise ValueError(
            f"t_batch shape {tuple(t_batch.shape)} != "
            f"grad_t shape {tuple(grad_t.shape)}"
        )
    if omega_batch.shape[-1] != 3:
        raise ValueError(
            f"last dim must be 3, got {tuple(omega_batch.shape)}"
        )

    R_current = exp_map_so3(omega_batch)         # (..., 3, 3)
    delta_R = exp_map_so3(step_size * grad_omega)  # (..., 3, 3)
    R_new = R_current @ delta_R                  # (..., 3, 3)
    omega_new = log_map_so3(R_new)               # (..., 3)
    t_new = t_batch + step_size * grad_t         # (..., 3)
    return omega_new, t_new
