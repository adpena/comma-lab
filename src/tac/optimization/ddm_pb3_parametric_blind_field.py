"""ddm_pb3 — a PARAMETRIC blind-set pose perturbation, in the coordinates it lives in.

WHY THIS EXISTS
---------------
``ddm_bp2`` proved (n600) that the 230,904 scorer-blind camera pixels are an exactly
seg-free pose actuator with 116x-154x reach, and then priced it out: naming the
coordinates costs more than the pose it buys (best case 2.32x under water at 668.81
signs-B/pair).  Its own named survivor was *"a PARAMETRIC blind-set perturbation --
one whose k coordinates and signs are generated from a handful of shipped scalars."*
That is this module.

THE DIAGNOSIS bp2's RECEIPTS ALREADY CONTAIN (DERIVED here, falsification-tested)
--------------------------------------------------------------------------------
``d_pose = ||e||^2 / 6`` with ``e = PoseNet(ours)[:6] - PoseNet(gt)[:6]`` -- a SIX
dimensional quadratic.  A perturbation ``delta`` acts through ``J = d p / d delta``
(6 x 692712), so::

    d_pose(delta) = || e + J delta ||^2 / 6

bp2 ran steepest descent in the 692,712-dim PIXEL space.  Its step direction
``-sign(g)``, ``g = (1/3) J^T e``, is only **~3-6% aligned** with the direction that
actually matters (``-e``), so >99.6% of its enormous reach lands in the five pose
dimensions that are already correct and it overshoots almost immediately.  The right
move is not a better ranking; it is to **SOLVE the 6-dimensional system**, which the
ENCODER can do (it has PoseNet at compress time) and then describe with a handful of
scalars.

DOF, JUSTIFIED RATHER THAN CHOSEN
---------------------------------
The target ``e`` is a 6-vector, so ``rank(J) <= 6``: no perturbation family can use
more than 6 output dimensions no matter how many of the 692,712 coordinates it
touches.  The optimal step is::

    delta* = -sign(g) = -sign( (J_pose^T e) pulled back through the warp )

Every weight in the composition ``D . (a M)`` is NON-NEGATIVE -- ``D`` is bilinear
downsampling (``1-frac``, ``frac``), ``M`` is bilinear resampling times a row-blend
``alpha in [0,1]`` -- so for ``a > 0`` the SIGN of ``g`` at a blind pixel is the sign
of the backprop image ``h = J_pose^T e`` at that pixel's warped location.  And
``h = sum_i e_i * (d p_i / d f0)`` lives in the **6-dimensional span** of the pose
saliency images.  So ``delta*`` is exactly:

    the sign of a 6-coefficient combination of six image fields, pulled back
    through the warp the receiver already builds.

That is the parametrization.  Six coefficients + one density = **7 scalars/pair**.

THE RECEIVER-COMPUTABLE BASIS (rule-118 FREE)
---------------------------------------------
``d p_i / d f0`` cannot be evaluated by the receiver (it needs PoseNet weights, which
never ship).  The direct visual-odometry ansatz supplies a generic stand-in: for any
estimator of a rigid twist from a frame pair, the photometric sensitivity of the
i-th twist component is a linear mix of the six **interaction-matrix projections**

    psi_i(x) = grad I(x) . L_i(x)

where ``L`` is the standard 2x6 interaction matrix at normalized image coordinates.
Only the SPAN matters, because the encoder fits the six coefficients freely -- any
invertible mixing (the estimator's Hessian inverse, unknown to us) is absorbed
exactly by the fit.  ``psi`` needs the render ``f0`` (the receiver makes it), the
intrinsics (a frozen constant), and an inverse-depth map from the ground plane the
v4d selector already carries.  **Zero counted bytes.**

WHAT SHIPS
----------
Per corrected pair: 6 direction coefficients + 1 density.  Nothing else -- no index,
no per-coordinate signs.  The encoder may decline any pair (c = 0), so the family is
one-sided safe.

AXIS
----
Everything this module computes is geometry and arithmetic.  Any ``d_pose`` number
produced with it is ``[macOS-CPU advisory]`` frozen-CPU-torch, ``score_claim=false``.
"""

from __future__ import annotations

from typing import Final

import numpy as np

# Rate arithmetic (upstream/evaluate.py:63 + the score formula).
CONTEST_RATE_DENOMINATOR: Final = 37_545_489
RATE_COEFF: Final = 25.0
N_PAIRS: Final = 600


# ------------------------------------------------------------------ arithmetic
def pose_contribution(d_pose_mean: float) -> float:
    """The score's pose term ``sqrt(10 * mean d_pose)``."""
    if d_pose_mean < 0.0:
        raise ValueError(f"d_pose must be non-negative, got {d_pose_mean}")
    return float(np.sqrt(10.0 * d_pose_mean))


def delta_s_rate(total_bytes: float) -> float:
    """Score cost of adding ``total_bytes`` to archive.zip."""
    return float(RATE_COEFF * total_bytes / CONTEST_RATE_DENOMINATOR)


def linearized_pose_floor(
    d_pose_base: np.ndarray,
    grad_blind_l1: np.ndarray,
    *,
    capture: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Lower bound on ``d_pose`` reachable by ANY 1-LSB blind perturbation.

    For any ``z`` in the reachable set ``Z = {J delta : ||delta||_inf <= 1}`` and any
    unit ``v``, ``||e + z|| >= <v,e> - h_Z(v)`` because ``Z`` is symmetric.  Taking
    ``v = e/||e||`` gives ``h_Z(v) = ||(J^T e)|blind||_1 / ||e|| = 3*g1/||e||`` where
    ``g1 = ||g_blind||_1`` is exactly bp2's recorded ``grad_blind_l1``.  With
    ``||e||^2 = 6 d``::

        d_floor / d  >=  (1 - g1 / (2 d))^2      when g1 < 2 d,   else 0

    ``capture`` in [0,1] is the fraction of ``g1`` a given perturbation family
    actually realizes (``capture = 1`` is the unconstrained optimum ``-sign(g)``;
    a parametric family realizes ``capture = eta`` per
    :func:`alignment_efficiency`).  This makes the bound a *family-specific* floor,
    not only a universal one.

    Valid under linearization of ``J``; the bound is falsification-tested against
    every measured arm in ``reports/ddm_bp2/reach_n600.jsonl``.
    """
    d = np.asarray(d_pose_base, dtype=np.float64)
    g1 = np.asarray(grad_blind_l1, dtype=np.float64)
    if np.any(d < 0) or np.any(g1 < 0):
        raise ValueError("d_pose and grad L1 must be non-negative")
    gamma = np.asarray(capture, dtype=np.float64) * g1 / np.maximum(2.0 * d, 1e-300)
    return d * np.clip(1.0 - gamma, 0.0, None) ** 2


def payload_bytes(n_pairs_corrected: int, scalars_per_pair: int, bits_per_scalar: int,
                  *, index_bytes: float = 0.0) -> float:
    """Total counted bytes for the parametric payload."""
    if min(n_pairs_corrected, scalars_per_pair, bits_per_scalar) < 0:
        raise ValueError("payload dimensions must be non-negative")
    return n_pairs_corrected * scalars_per_pair * bits_per_scalar / 8.0 + index_bytes


def subset_index_bytes(n_selected: int, n_total: int = N_PAIRS) -> float:
    """Combinatorial (colex-rank) cost of naming WHICH pairs carry a correction."""
    if not 0 <= n_selected <= n_total:
        raise ValueError(f"n_selected {n_selected} outside [0, {n_total}]")
    if n_selected in (0, n_total):
        return 0.0
    from math import lgamma, log

    log2c = (
        lgamma(n_total + 1) - lgamma(n_selected + 1) - lgamma(n_total - n_selected + 1)
    ) / log(2.0)
    return log2c / 8.0


# -------------------------------------------------------- receiver-side basis
def normalized_grid(height: int, width: int, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalized image coordinates ``x=(u-cx)/fx``, ``y=(v-cy)/fy``."""
    fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    u = np.arange(width, dtype=np.float64)[None, :]
    v = np.arange(height, dtype=np.float64)[:, None]
    return (u - cx) / fx, (v - cy) / fy


def ground_inverse_depth(y_norm: np.ndarray) -> np.ndarray:
    """``1/Z`` for a ground plane, in camera-height units: ``1/Z = y`` for ``y > 0``.

    A ground point at normalized row ``y`` below the horizon sits at ``Z = h / y``,
    so ``1/Z = y / h``; the camera height ``h`` is a pure scale that the coefficient
    fit absorbs, so it is set to 1 and never shipped.  Above the horizon the ground
    model does not apply and the point is treated as at infinity (``1/Z = 0``) --
    the same far/ground split the v4d selector already carries.
    """
    return np.clip(y_norm, 0.0, None)


def interaction_matrix(x_norm: np.ndarray, y_norm: np.ndarray, inv_z: np.ndarray) -> np.ndarray:
    """The standard 2x6 image-motion interaction matrix, broadcast over the grid.

    Returns ``(2, 6, H, W)``: row 0 is ``d x / d twist``, row 1 is ``d y / d twist``,
    twist ordered ``(vx, vy, vz, wx, wy, wz)``.
    """
    x, y = np.broadcast_arrays(x_norm, y_norm)
    z = np.broadcast_to(inv_z, x.shape)
    zero = np.zeros_like(x)
    one = np.ones_like(x)
    lx = np.stack([-z, zero, x * z, x * y, -(one + x * x), y])
    ly = np.stack([zero, -z, y * z, one + y * y, -x * y, -x])
    return np.stack([lx, ly])


def vo_saliency_fields(f0_hwc: np.ndarray, k: np.ndarray) -> np.ndarray:
    """The six receiver-computable pose-saliency fields ``psi_i = grad I . L_i``.

    ``(6, H, W, C)``.  Gradients are central differences in NORMALIZED coordinates
    (``d/dx = fx * d/du``), so the interaction matrix and the gradient share units.
    Deterministic; depends only on the render, the frozen intrinsics, and the ground
    model -- nothing video-derived is stored.
    """
    if f0_hwc.ndim != 3:
        raise ValueError(f"f0 must be (H,W,C), got {f0_hwc.shape}")
    height, width, _ = f0_hwc.shape
    img = np.asarray(f0_hwc, dtype=np.float64)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    gu = np.gradient(img, axis=1) * fx
    gv = np.gradient(img, axis=0) * fy
    x_norm, y_norm = normalized_grid(height, width, k)
    lmat = interaction_matrix(x_norm, y_norm, ground_inverse_depth(y_norm))
    return lmat[0][..., None] * gu[None] + lmat[1][..., None] * gv[None]


def random_polynomial_saliency_fields(
    f0_hwc: np.ndarray, k: np.ndarray, rng: np.random.Generator, *, n_fields: int = 6
) -> np.ndarray:
    """CONTROL basis: ``grad I . (random quadratic)`` — the same functional class.

    Every entry of the interaction matrix is a polynomial of degree <= 2 in the
    normalized coordinates, so a basis built from RANDOM quadratics has the same
    rank, the same smoothness, and the same ``grad I`` factor.  If it explains the
    true gradient as well as :func:`vo_saliency_fields` does, then the ego-motion
    geometry is not what carries the signal — only the rank and the image gradient
    are — and the design's central claim is refuted.  A white-noise control would be
    trivially uninformative; this one is not.
    """
    height, width, _ = f0_hwc.shape
    img = np.asarray(f0_hwc, dtype=np.float64)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    gu = np.gradient(img, axis=1) * fx
    gv = np.gradient(img, axis=0) * fy
    x_norm, y_norm = normalized_grid(height, width, k)
    x, y = np.broadcast_arrays(x_norm, y_norm)
    monomials = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y])
    out = []
    for _ in range(n_fields):
        p = rng.standard_normal(monomials.shape[0])
        q = rng.standard_normal(monomials.shape[0])
        out.append(
            np.einsum("i,ihw->hw", p, monomials)[..., None] * gu
            + np.einsum("i,ihw->hw", q, monomials)[..., None] * gv
        )
    return np.stack(out)


def pullback_to_blind(
    idx: np.ndarray, w: np.ndarray, a: float, fields: np.ndarray, blind: np.ndarray
) -> np.ndarray:
    """Pull scorer-side camera fields back onto the blind frame_1 coordinates.

    ``fields`` ``(m,H,W,C)`` -> ``(m, n_blind)``.  This is the SAME adjoint bp2 uses
    to turn a frame_0 cotangent into a frame_1 sensitivity, so the basis lives in
    exactly the coordinates the true gradient does.
    """
    from tac.optimization.ddm_bp2_blind_pose_actuator import adjoint_taps

    return np.stack([adjoint_taps(idx, w, a * f)[blind].ravel() for f in fields])


# ---------------------------------------------------------------- the fit
def fit_basis_coefficients(
    g_blind: np.ndarray, basis: np.ndarray
) -> tuple[np.ndarray, float]:
    """Least-squares fit of the true blind gradient onto the basis span.

    Returns ``(c, r2)``.  ``r2`` is the fraction of the true gradient's energy the
    receiver-computable span explains -- the honest single number for "does the
    ansatz hold".  Any invertible mixing of the basis is absorbed by ``c``, which is
    why only the SPAN of the saliency fields has to be right.
    """
    g = np.asarray(g_blind, dtype=np.float64).ravel()
    b = _checked_basis(basis, g.size)
    c, *_ = np.linalg.lstsq(b.T, g, rcond=None)
    resid = g - b.T @ c
    denom = float(g @ g)
    r2 = 0.0 if denom == 0.0 else float(1.0 - (resid @ resid) / denom)
    return c, r2


def _checked_basis(basis: np.ndarray, n: int) -> np.ndarray:
    """Column-normalized basis, with a fail-LOUD non-finite guard.

    Normalizing rescales the coefficients but not the SPAN, so nothing about the
    family changes; it stops an ill-conditioned control basis from producing ``inf``
    coefficients whose ``sign`` is ``nan`` — a wrong answer that never raises.
    """
    b = np.asarray(basis, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] != n:
        raise ValueError(f"basis {b.shape} does not match gradient length {n}")
    if not np.all(np.isfinite(b)):
        raise ValueError("basis contains non-finite values")
    scale = np.linalg.norm(b, axis=1, keepdims=True)
    return b / np.where(scale > 0.0, scale, 1.0)


def fit_max_alignment(
    g_blind: np.ndarray,
    basis: np.ndarray,
    *,
    n_restarts: int = 512,
    n_search: int = 60_000,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Maximize ``eta`` DIRECTLY over the coefficient sphere, not via least squares.

    ``eta(c) = sum_j g_j sign(B_j . c) / ||g||_1`` is scale-invariant in ``c`` and
    piecewise constant, so the L2 fit of :func:`fit_basis_coefficients` is only a
    heuristic for it — and reporting an L2-fitted ``eta`` as the family's capacity
    would be a strawman of exactly the kind bp2's own FORMULATION-scoped negative
    warns against.  Search: random directions on ``S^(m-1)`` plus the L2 solution and
    each single-field axis, scored on an importance subsample (drawn without
    replacement, ``|g|``-weighted, because ``eta`` is a ``|g|``-weighted statistic),
    then the best candidates re-scored on ALL coordinates.

    Returns ``(c, eta)`` with ``eta`` evaluated on the FULL coordinate set.
    """
    g = np.asarray(g_blind, dtype=np.float64).ravel()
    b = _checked_basis(basis, g.size)
    rng = np.random.default_rng(seed)
    l1 = float(np.abs(g).sum())
    if l1 == 0.0:
        return np.zeros(b.shape[0]), 0.0

    cands = [rng.standard_normal((n_restarts, b.shape[0]))]
    cands.append(np.eye(b.shape[0]))
    cands.append(-np.eye(b.shape[0]))
    c_ls, _ = fit_basis_coefficients(g, b)
    cands.append(c_ls[None, :])
    cand = np.concatenate(cands)

    if n_search < g.size:
        p = np.abs(g)
        p = p / p.sum()
        sub = rng.choice(g.size, size=n_search, replace=False, p=p)
        gs, bs = g[sub], b[:, sub]
    else:
        gs, bs = g, b
    coarse = (np.sign(cand @ bs) * gs).sum(axis=1)
    keep = cand[np.argsort(-coarse)[: min(16, cand.shape[0])]]

    best_c, best_eta = keep[0], -np.inf
    for c in keep:
        eta = float((np.sign(b.T @ c) * g).sum() / l1)
        if eta > best_eta:
            best_c, best_eta = c, eta
    return best_c, best_eta


def alignment_efficiency(
    g_blind: np.ndarray, phi: np.ndarray, *, density: float = 1.0
) -> tuple[float, np.ndarray]:
    """``eta`` = fraction of the maximum first-order pose descent this field realizes.

    The step is ``delta = -sign(phi)`` on the ``density`` fraction of coordinates with
    the largest ``|phi|``, zero elsewhere -- the realization that matches the true
    optimum's structure (``delta* = -sign(g)``).  Then::

        first-order  d(d_pose) = <g, delta> = -eta * ||g_blind||_1

    ``eta = 1`` is the unconstrained optimum, ``eta = 0`` is inert, ``eta < 0`` is an
    ascent (the encoder would decline the pair).  Feeds :func:`linearized_pose_floor`
    as ``capture``.
    """
    if not 0.0 < density <= 1.0:
        raise ValueError(f"density must be in (0,1], got {density}")
    g = np.asarray(g_blind, dtype=np.float64).ravel()
    p = np.asarray(phi, dtype=np.float64).ravel()
    if g.shape != p.shape:
        raise ValueError(f"shape mismatch {g.shape} vs {p.shape}")
    if not np.all(np.isfinite(p)):
        raise ValueError("phi contains non-finite values — sign() would be nan")
    delta = -np.sign(p)
    if density < 1.0:
        keep = int(max(1, round(density * p.size)))
        cut = np.partition(np.abs(p), p.size - keep)[p.size - keep]
        delta = np.where(np.abs(p) >= cut, delta, 0.0)
    l1 = float(np.abs(g).sum())
    eta = 0.0 if l1 == 0.0 else float(-(g @ delta) / l1)
    return eta, delta
