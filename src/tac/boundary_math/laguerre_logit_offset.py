# SPDX-License-Identifier: MIT
"""Margin-field HEAD levers (#218 facets 1 & 3) — BYTE-FREE per-class logit geometry.

The level-set witness renders ``argmax_k phi_k`` (K=5 SDF fields) into a palette
partition that the frozen SegNet re-reads as its argmax (the REALIZED ``d_seg``).
The measured binding residual is the ERASURE long tail: the minority classes
(``1=Lane`` 0.59% area, ``3=Movable`` 1.56%) are systematically dropped below the
argmax margin — Lane<->Road alone is ~57% of all flips (#209). This is the
canonical **class-imbalance under-prediction** failure of a plain-softmax head.

This module implements the three HEAD levers that fix it at **ZERO archive bytes**:

* **Facet 3 — per-class LOGIT ADJUSTMENT (Menon et al., 2007.07314).** A per-class
  ADDITIVE offset ``b_k`` on the SDF logits: the decoded partition becomes
  ``argmax_k (phi_k + b_k)``.  Because ``phi_k = W_k . h + bias_k``, the offset
  folds directly into the *already-counted* ``out_sdf.bias`` (5 floats) — the
  archive size is UNCHANGED, so the lever is byte-free.  Geometrically this is a
  **power diagram (Laguerre) reweighting** of the witness's Voronoi/argmax cells:
  ``argmax_k phi_k`` is a (generalized) nearest-prototype diagram; adding
  per-class weights ``b_k`` is exactly the Laguerre weight that shifts each cell's
  boundary (``argmax-of-SDF == power diagram``).  Boosting a rare class (``b_k>0``)
  enlarges its cell, recovering erased Lane/Movable pixels.

* **Facet 1a — fixed simplex-ETF head (Yang et al. 2022; neural-collapse optimal).**
  Replace the *learned* classifier ``out_sdf.weight`` (K x d) with a FIXED simplex
  Equiangular Tight Frame: ``||w_k|| == const`` and ``<w_i, w_j> == -1/(K-1)`` for
  all ``i != j`` — the max-equiangular configuration neural collapse converges to.
  Fixing it removes the minority-class *norm collapse* (rare classes otherwise get
  small-norm prototypes -> weak logits -> erasure) AND, being deterministically
  regenerable from a seed, makes the K x d head weight FREE at inflate (a small
  RATE win on top of the d_seg win).

* **Facet 1b — additive-margin softmax (Wang et al. 2018, CosFace / AM-Softmax).**
  Train with a margin ``m`` subtracted from the TARGET-class logit:
  ``softmax(phi_y - m, phi_{k!=y})``.  This enlarges the inter-class SDF margin so
  boundary pixels sit further from the flip locus and SURVIVE the ``R`` round-trip
  (bicubic^ -> uint8 -> bilinear_).  Training-time only: 0 archive bytes.

AUTHORITY: this is a HEAD-DESIGN / loss-weighting asset, never a score claim.  All
numpy here is the bit-identical fp32/fp64 reference; the MLX twins are one-liners
the trainer inlines (``mx.softmax``/``argmax`` over the same tensors).  The only
authoritative ``d_seg`` is ``cpu_verdict_d_seg_batch`` (frozen CPU SegNet through R)
on a byte-closed render; ``laguerre_offset_sweep`` returns the TASK-SPACE argmax
disagreement (a fast ranking proxy for realized d_seg — the witness palette is
near-hard at the annealed temperature, so phi-argmax tracks the SegNet argmax).
NO FAKE: the sweep runs on the REAL witness phi field; the winner MUST be
re-confirmed through R before any promotion.

Rides (does not duplicate):
  * ``tac.margin_saliency_map`` (#141) — the frozen-SegNet ``d margin / d input``
    producer (the saliency that weights facet-1b's margin).
  * ``tac.boundary_math.margin_polytope`` — the per-pixel free-budget
    ``b(p) = m(p)/||g_p||`` (the polytope radius the offset must exceed to flip).
  * ``tac.boundary_math.lever_b_levelset_generator`` — the witness head
    (``out_sdf`` -> phi -> palette) these levers modify.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Canonical comma10k SegNet class order (SELF-DETECT elsewhere; here it is only a
# label vocabulary for reporting — the math is index-agnostic).
CLASS_NAMES: tuple[str, ...] = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
_ETF_SEED = 218  # #218 facet id — a FIXED seed so the ETF head is deterministically regenerable (FREE at inflate)


class LaguerreLogitOffsetError(ValueError):
    """Raised on malformed head-lever inputs."""


# ---------------------------------------------------------------------------
# Facet 3 — the Laguerre / power-diagram per-class additive offset
# ---------------------------------------------------------------------------
def power_diagram_argmax(phi: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """``argmax_k (phi_k + b_k)`` — the Laguerre-reweighted partition.

    ``phi``: ``(..., K)`` SDF/logit field.  ``offsets``: ``(K,)`` Laguerre weights.
    Returns the argmax label field ``(...)`` (int64).  With ``offsets == 0`` this is
    exactly ``argmax_k phi_k`` (the witness's un-reweighted partition).
    """
    phi = np.asarray(phi, dtype=np.float64)
    b = np.asarray(offsets, dtype=np.float64).reshape(-1)
    if phi.shape[-1] != b.shape[0]:
        raise LaguerreLogitOffsetError(
            f"offsets K={b.shape[0]} != phi last dim K={phi.shape[-1]}"
        )
    return np.argmax(phi + b, axis=-1).astype(np.int64)


def apply_offset_to_sdf_bias(
    params: dict[str, np.ndarray], offsets: np.ndarray, *, bias_key: str = "out_sdf.bias"
) -> dict[str, np.ndarray]:
    """Fold the per-class offset into ``out_sdf.bias`` (the BYTE-FREE application).

    Returns a SHALLOW copy of ``params`` with ``bias_key`` replaced by
    ``bias + offsets``; the input dict and its arrays are NOT mutated.  Because the
    bias is already a counted parameter, this changes its VALUE not its SIZE ->
    0 extra archive bytes.  The identity ``phi(bias+b) == phi(bias) + b`` makes this
    equivalent to :func:`power_diagram_argmax` at decode.
    """
    if bias_key not in params:
        raise LaguerreLogitOffsetError(f"params has no '{bias_key}'")
    bias = np.asarray(params[bias_key], dtype=np.float32)
    b = np.asarray(offsets, dtype=np.float32).reshape(-1)
    if b.shape[0] != bias.shape[0]:
        raise LaguerreLogitOffsetError(
            f"offsets K={b.shape[0]} != bias K={bias.shape[0]}"
        )
    out = dict(params)
    out[bias_key] = (bias + b).astype(np.float32)
    return out


def menon_logit_adjustment_offsets(
    class_priors: np.ndarray, *, tau: float = 1.0, eps: float = 1e-12
) -> np.ndarray:
    """Menon (2007.07314) DECODE-time per-class offset ``b_k = -tau * log(pi_k)``.

    ``class_priors``: ``(K,)`` non-negative class frequencies (need not be
    normalized).  Rare classes (small ``pi_k``) get a LARGER positive offset ->
    their argmax cell is enlarged -> systematic under-prediction is corrected.  The
    offsets are mean-centered (a global constant does not change any argmax) so the
    result is the canonical zero-sum Laguerre weight vector.  Use as the INIT for
    the trainable per-class bias (facet 3) or directly as a decode offset.
    """
    p = np.asarray(class_priors, dtype=np.float64).reshape(-1)
    if p.ndim != 1 or p.shape[0] < 2:
        raise LaguerreLogitOffsetError("class_priors must be (K,) with K>=2")
    if (p < 0).any():
        raise LaguerreLogitOffsetError("class_priors must be non-negative")
    total = float(p.sum())
    if total <= 0:
        raise LaguerreLogitOffsetError("class_priors sum must be > 0")
    pi = p / total
    b = -float(tau) * np.log(np.maximum(pi, eps))
    b = b - b.mean()  # zero-sum: argmax is invariant to a global constant
    return b.astype(np.float64)


# ---------------------------------------------------------------------------
# Facet 3b — damped-Newton semi-discrete OT offset (the deep-math "Amortizing the
# Argmax" Ch.1 tropical/Laguerre lens; Kitagawa-Merigot-Thibert 2019). The
# PRINCIPLED solver that REPLACES the Menon -tau*log(pi) heuristic: it SOLVES the
# exact per-class offset b* whose Laguerre-reweighted cell MASSES equal the target
# frequencies, accounting for THIS witness's logit geometry (boundary lengths) that
# the log-freq heuristic ignores. Byte-free (K offsets); apply via
# apply_offset_to_sdf_bias. Attacks the minority-collapse ASYMMETRY at its root.
# ---------------------------------------------------------------------------
def soft_cell_masses(phi: np.ndarray, offsets: np.ndarray, *, tau: float = 1.0) -> np.ndarray:
    """Mean soft (``softmax_tau``) class masses of the Laguerre-reweighted field.

    ``m_c(b) = mean_p softmax((phi_p + b)/tau)_c`` -- smooth in ``b``; the hard
    ``tau -> 0`` limit is the :func:`power_diagram_argmax` cell-mass fraction. Returns
    ``(K,)`` summing to 1. Numerically stable (row-max subtracted before exp).
    """
    z = np.asarray(phi, dtype=np.float64)
    z = z.reshape(-1, z.shape[-1]) + np.asarray(offsets, dtype=np.float64).reshape(-1)
    z = z / max(float(tau), 1e-9)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    s = e / np.maximum(e.sum(axis=1, keepdims=True), 1e-300)
    return s.mean(axis=0)


def hard_cell_masses(phi: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Mean HARD (argmax) Laguerre cell masses ``m_c = mean_p [argmax(phi_p+b)==c]``
    -- the ``tau -> 0`` limit of :func:`soft_cell_masses`; the quantity the offset is
    solved to match. Returns ``(K,)`` summing to 1."""
    lab = power_diagram_argmax(phi, offsets)
    k = int(np.asarray(phi).shape[-1])
    return np.bincount(lab.reshape(-1), minlength=k).astype(np.float64) / float(lab.size)


def softmax_cov_condition_number(
    cov: np.ndarray, *, gauge_dim: int = 1, eps_rel: float = 1e-12,
) -> float:
    """Range condition number ``lambda_max / lambda_min`` of the softmax-covariance
    Hessian ``cov = diag(m) - mean_p s_p s_p^T``, EXCLUDING the ``gauge_dim``-dim
    all-ones gauge nullspace (the smallest ``gauge_dim`` eigenvalues, ``~0`` by
    construction). A near-free anisotropy sensor (Nielsen structure-tensor 2307.10644):
    the boundary-annulus Hessian is anisotropic (flat-interior + sharp-along-boundary,
    #333); a HIGH range-condition-number means the Newton solve is genuinely
    ill-conditioned (preconditioning pays); ``~1`` means it is already well-conditioned
    (the eigendecomposition would only add cost). Eigenvalues below ``eps_rel*lambda_max``
    are treated as numerical-zero (part of the effective nullspace). Returns ``+inf`` if
    the effective range is empty (rank <= gauge_dim). ``cov`` must be symmetric PSD.
    """
    w = np.linalg.eigvalsh(np.asarray(cov, dtype=np.float64))  # ascending, real
    w = np.clip(w, 0.0, None)
    rng = w[int(gauge_dim):]  # the K-gauge_dim eigenvalues that a full-rank solve expects positive
    if rng.size == 0:
        return float("inf")
    lam_max, lam_min = float(rng[-1]), float(rng[0])
    if lam_max <= 0.0 or lam_min <= eps_rel * lam_max:
        # rank-deficient IN the expected range (a class-absent / saturated direction) =>
        # the Newton solve is degenerate there => maximally ill-conditioned. Do NOT filter
        # the small eigenvalue out: lambda_min -> 0 IS the ill-conditioning we are sensing.
        return float("inf")
    return float(lam_max / lam_min)


def _newton_step_from_cov(
    cov: np.ndarray, g: np.ndarray, taus: float, *,
    precondition: bool, rcond: float, eps_rel: float, cond_gate: float | None,
) -> tuple[np.ndarray, float]:
    """The dual-ascent Newton direction ``taus * H^+ @ g`` (``H = -cov/taus``), returned
    with the range condition number of ``cov``.

    ``precondition=False`` -> the EXACT legacy path ``taus * pinv(cov, rcond) @ g``
    (byte-identical to the pre-2026-07-10 solver; the preconditioner is opt-in only).

    ``precondition=True`` -> Hessian-preconditioned conjugation (Plus-Gourdon & Nielsen
    2606.09077): eigendecompose ``cov = Q diag(evals) Q^T``, deform to the canonical
    paraboloid (whiten: the well-conditioned residual is solved in ``Q``-coordinates where
    the Hessian is diagonal), invert only the eigenvalues above the relative floor
    ``eps_rel*lambda_max`` (explicitly dropping the all-ones gauge nullspace — its eigenvalue
    is ``~0`` << the floor), then map back: ``step = taus * Q (evals^+ (Q^T g))``. NOTE: for a
    single DENSE solve this is algebraically the same Newton step as ``np.linalg.pinv``
    (whose ``rcond`` is ALSO a floor relative to the largest singular value); the differences
    are (a) the explicit eigenbasis exposes the range condition number as a near-free
    by-product and (b) the floor is gauge-explicit. The paper's convergence speedup targets
    ITERATIVE inner solves (CG / learned conjugation), NOT a dense pinv — so on this 5x5
    solve preconditioning buys robustness + the sensor, not iterations (MEASURED: see the A/B
    probe ``experiments/probe_hessian_precond_ot_ab.py``).

    ``cond_gate`` (structure-tensor gate): when not ``None`` and the range condition number
    is BELOW it, fall through to the fast legacy ``pinv`` even if ``precondition=True`` — the
    eigendecomposition does not pay on an already-well-conditioned Hessian. This answers
    "WHERE does preconditioning pay" with the ``lambda_max/lambda_min`` the eigendecomp
    already exposes.
    """
    cond = softmax_cov_condition_number(cov, eps_rel=eps_rel)
    if (not precondition) or (cond_gate is not None and cond < float(cond_gate)):
        return taus * (np.linalg.pinv(cov, rcond=rcond) @ g), cond
    evals, q = np.linalg.eigh(np.asarray(cov, dtype=np.float64))  # ascending, real (PSD)
    lam_max = max(float(evals[-1]), 1e-300)
    inv = np.where(evals > eps_rel * lam_max, 1.0 / np.maximum(evals, 1e-300), 0.0)
    step = taus * (q @ (inv * (q.T @ np.asarray(g, dtype=np.float64))))
    return step, cond


def damped_newton_ot_offsets(
    phi: np.ndarray, target_masses: np.ndarray, *, tau: float = 1.0,
    max_iter: int = 64, tol: float = 1e-10, rcond: float = 1e-10,
    precondition: bool = False, precond_eps_rel: float = 1e-9,
    precond_cond_gate: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Damped-Newton semi-discrete OT solve for the zero-sum per-class offset ``b*``
    with ``soft_cell_masses(phi, b*, tau) == target_masses`` (Kitagawa-Merigot-Thibert
    2019; the hard ``tau -> 0`` limit is the Aurenhammer-Hoffmann-Aronov power-diagram
    weight). A REAL Newton solve (NOT a sweep):

      dual   Phi(b) = <pi, b> - tau * mean_p logsumexp((phi_p + b)/tau)   (concave)
      grad   g(b)   = pi - m(b)                                            (m = soft masses)
      Hess   H(b)   = -(1/tau) * (diag(m) - mean_p s_p s_p^T)             (NSD softmax cov)
      step   b <- b + t * pinv(diag(m) - mean_p s_p s_p^T) @ (pi - m) * tau

    ``H`` is rank ``K-1`` (all-ones gauge nullspace) => zero-sum pseudo-inverse
    (``rcond``); ``t`` is backtracked (halved) until the dual increases (Armijo), which
    guarantees global convergence; the terminal rate is quadratic. ``phi``: ``(...,K)``
    witness SDF/logit field; ``target_masses``: ``(K,)`` non-negative (renormalized).
    Returns ``(b*, info)`` with ``info`` = {converged, iters, max_mass_err, dual,
    cond_number (range lambda_max/lambda_min at the final iterate), preconditioned}.
    Byte-free: fold ``b*`` into the bias via :func:`apply_offset_to_sdf_bias`.

    ``precondition`` (opt-in; default ``False`` == byte-identical legacy pinv path):
    compute each Newton step via Hessian-preconditioned conjugation (Plus-Gourdon &
    Nielsen 2606.09077) through :func:`_newton_step_from_cov` — eigendecompose the
    softmax-covariance Hessian, whiten to the canonical paraboloid, invert eigenvalues
    above the RELATIVE floor ``precond_eps_rel*lambda_max`` (gauge-explicit), map back.
    ``precond_cond_gate`` (structure-tensor gate): when set, use the preconditioner only
    where the range condition number exceeds it (else the fast legacy pinv). Both paths
    solve the SAME concave dual to the SAME fixed point ``b*``; preconditioning changes
    only the per-step numerics/robustness, never the objective (NO-FAKE: the through-R
    d_seg is identical when ``b*`` is identical, which the A/B measures).
    """
    z0 = np.asarray(phi, dtype=np.float64)
    z0 = z0.reshape(-1, z0.shape[-1])
    n, k = z0.shape
    pi = np.asarray(target_masses, dtype=np.float64).reshape(-1)
    if pi.shape[0] != k:
        raise LaguerreLogitOffsetError(f"target_masses K={pi.shape[0]} != phi K={k}")
    if (pi < 0).any() or float(pi.sum()) <= 0:
        raise LaguerreLogitOffsetError("target_masses must be non-negative with sum>0")
    pi = pi / float(pi.sum())
    taus = max(float(tau), 1e-9)
    phi_c = z0 / taus  # (N,K), the b-independent part of z/tau

    b = np.zeros(k, dtype=np.float64)

    # concave dual  Phi(b) = <pi,b> - tau*mean_p LSE(phi_c + b/tau),  phi_c := phi/tau
    def _dual(bb: np.ndarray) -> float:
        zz = phi_c + (bb / taus)
        mm = zz.max(axis=1, keepdims=True)
        lse = mm[:, 0] + np.log(np.exp(zz - mm).sum(axis=1))
        return float(np.dot(pi, bb) - taus * lse.mean())

    info = {"converged": 0.0, "iters": 0.0, "max_mass_err": 1.0, "dual": _dual(b),
            "cond_number": float("nan"), "preconditioned": float(bool(precondition))}
    for it in range(1, int(max_iter) + 1):
        zz = phi_c + (b / taus)
        zz = zz - zz.max(axis=1, keepdims=True)
        e = np.exp(zz)
        s = e / np.maximum(e.sum(axis=1, keepdims=True), 1e-300)   # (N,K) softmax
        m = s.mean(axis=0)                                          # (K,) soft masses
        g = pi - m                                                 # gradient of the dual
        err = float(np.max(np.abs(g)))
        info.update(iters=float(it), max_mass_err=err, dual=_dual(b))
        if err <= tol:
            info["converged"] = 1.0
            break
        cov = np.diag(m) - (s.T @ s) / float(n)                    # (K,K) softmax covariance (PSD, rank K-1)
        step, cond = _newton_step_from_cov(                        # Newton direction (dual ascent)
            cov, g, taus, precondition=precondition, rcond=rcond,
            eps_rel=precond_eps_rel, cond_gate=precond_cond_gate)
        info["cond_number"] = cond                                 # range lambda_max/lambda_min at this iterate
        step = step - step.mean()                                  # stay zero-sum (gauge)
        # Backtracking on the concave dual: accept the largest t in {1,1/2,...} whose
        # step does NOT DECREASE Phi. Near the optimum the ascent direction's full
        # (t=1) Newton step barely moves Phi (flat), so a non-decrease criterion (not
        # strict-increase) is what preserves terminal QUADRATIC convergence.
        t, base = 1.0, _dual(b)
        for _ in range(60):
            if _dual(b + t * step) >= base - 1e-15 * max(1.0, abs(base)):
                break
            t *= 0.5
        b = b + t * step
        b = b - b.mean()
    return b.astype(np.float64), info


# ---------------------------------------------------------------------------
# Facet 3c — FLIP-WEIGHTED head offsets (crucible-3 N-1 reformulation, task #386). N-1
# MEASURED (n600) that OT AREA-mass-matching to GT class FREQUENCIES HURTS realized d_seg
# (no_offset 0.0031436 < menon 0.0033119 < ot_newton 0.0048921; verdict_scope FORMULATION —
# the SOLVER is exact, the OBJECTIVE was wrong). The reformulation targets the FLIP mass the
# scorer actually re-reads (the codim-1 boundary annulus, #333), NOT the bulk cell area, in
# two independent formulations that the $0 n600 3-arm gate arbitrates:
#
#   * ``flip_weighted`` — the SAME damped-Newton OT solve, but the target masses are the
#     per-class FLIP SHARE (``flips_c / total_flips``, the canonical ``perclass_verdict.
#     flip_share_by_class`` sensor) instead of GT area frequency. This is an UN-ANALYZED
#     objective (crucible-3 P3 F3): OT still mass-MATCHES cells, so it may re-inherit N-1's
#     cell-inflation pathology; the through-R gate is the decisive arbiter.
#   * ``flip_median`` — S1's Hamming-OPTIMAL per-edge threshold. d_seg is HAMMING (0-1), whose
#     L1-optimal 1-D threshold is the flip-density MEDIAN, NOT a Wasserstein mass-match. This is
#     a DISTINCT closed-form solve path (NOT expressible through the OT target-mass machinery —
#     no median/quantile solver existed; P3 F3 confirmed). See ``flip_median_offsets``.
#
# Both do REAL work on REAL inputs (NO-FAKE); both fold BYTE-FREE into ``out_sdf.bias``.
# ---------------------------------------------------------------------------
def _flip_share_by_class(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> np.ndarray:
    """Per-class FLIP SHARE ``flips_c / total_flips`` (which GT class CARRIES the residual).

    Delegates to the canonical sensor ``tac.witness_control.perclass_verdict.per_class_flip_stats``
    (the SAME ``flip_share_by_class`` the #315 per-class λ costate reads) — lazy-imported so
    ``boundary_math`` stays import-cycle-free. Returns ``(K,)`` summing to 1. RAISES (never returns a
    silent all-zero target the OT solve would choke on) when the witness argmax has ZERO flips vs GT.
    """
    from tac.witness_control.perclass_verdict import per_class_flip_stats

    p = np.asarray(pred).reshape(-1)
    g = np.asarray(gt).reshape(-1)
    flips, _pixels = per_class_flip_stats([p], [g], n_classes=int(num_classes))
    total = float(flips.sum())
    if total <= 0:
        raise LaguerreLogitOffsetError(
            "flip_share undefined: the witness argmax has ZERO flips vs GT (no residual to target)")
    return flips.astype(np.float64) / total


def flip_median_offsets(
    phi: np.ndarray, gt: np.ndarray, *, pred: np.ndarray | None = None,
    num_classes: int | None = None, edge_min_flips: int = 1,
) -> tuple[np.ndarray, dict[str, float]]:
    """S1's HAMMING-optimal per-edge threshold: ``b_c - b_{c'} = -MEDIAN`` over the edge's FLIP
    pixels of the margin ``m = phi_c - phi_{c'}`` (crucible-3 N-1 reformulation, task #386).

    d_seg is a HAMMING (0-1) loss, whose L1-optimal 1-D threshold along a margin is the MEDIAN of
    the boundary-margin distribution — NOT the Wasserstein area-match N-1 MEASURED as a NEGATIVE
    (no_offset 0.00314 < ot_newton 0.00489 n600). This is a DISTINCT closed-form solve path, NOT a
    target-mass choice fed to the OT machinery (which has no median/quantile solver — crucible-3 P3
    F3). Per unordered edge ``{i,j}`` the FLIP pixels are those whose ``{gt, pred}`` is exactly
    ``{i,j}`` with ``gt != pred`` (GT=i wrongly predicted j, or GT=j wrongly predicted i); the offset
    difference that moves the decode threshold to the flip-margin median is
    ``delta_ij = b_i - b_j = -median(m over those flips)``.

    ``pred`` is the flip-identifying prediction. Pass the **REALIZED** argmax (the frozen-SegNet
    argmax on the rendered frame — the actual d_seg residual we minimise); if omitted it falls back
    to ``argmax(phi)`` (the un-rendered phi-space proxy, which for this witness disagrees with the
    realized SegNet argmax by ~50x, so the realized ``pred`` is strongly preferred). The margins are
    always the phi-space ``phi_i - phi_j`` (the offset lives in phi-space); only the flip SET is
    ``pred``-defined.

    The per-edge deltas over-determine the K-vector (up to ``C(K,2)`` edges, ``K-1`` free DOF), so
    they are reconciled by a flip-count-WEIGHTED least squares on the edge-difference graph with the
    zero-sum gauge: solve ``L b = r`` where ``L`` is the flip-count-weighted graph Laplacian and
    ``r_i = sum_j w_ij * (+/-)delta_ij`` (``+`` for the lower edge index). ``L`` is rank ``K-1``
    (all-ones nullspace) => pseudo-inverse gives the min-norm (zero-sum) solution. Byte-free: fold
    ``b*`` into ``out_sdf.bias`` via :func:`apply_offset_to_sdf_bias`.

    ``phi``: ``(..., K)`` witness SDF/logit field. ``gt``: ``(...)`` GT argmax labels. With NO flips
    (perfect witness) returns ``b == 0`` (the correct no-op). Returns ``(b*, info)`` with ``info`` =
    ``{converged, iters, max_mass_err (NaN — mass is NOT the objective), n_edges_used, total_flips,
    pred_is_realized}``.
    """
    z = np.asarray(phi, dtype=np.float64)
    z = z.reshape(-1, z.shape[-1])
    n, k = z.shape
    if num_classes is not None and int(num_classes) != k:
        raise LaguerreLogitOffsetError(f"num_classes {num_classes} != phi K {k}")
    if k < 2:
        raise LaguerreLogitOffsetError("phi must have K>=2 classes")
    g = np.asarray(gt).reshape(-1).astype(np.int64)
    if g.shape[0] != n:
        raise LaguerreLogitOffsetError(f"gt size {g.shape[0]} != phi rows {n}")
    if g.size and (int(g.min()) < 0 or int(g.max()) >= k):
        raise LaguerreLogitOffsetError("gt labels out of range [0,K)")

    if pred is None:
        pred0 = np.argmax(z, axis=1)      # phi-space fallback decode
        pred_is_realized = 0.0
    else:
        pred0 = np.asarray(pred).reshape(-1).astype(np.int64)
        if pred0.shape[0] != n:
            raise LaguerreLogitOffsetError(f"pred size {pred0.shape[0]} != phi rows {n}")
        if pred0.size and (int(pred0.min()) < 0 or int(pred0.max()) >= k):
            raise LaguerreLogitOffsetError("pred labels out of range [0,K)")
        pred_is_realized = 1.0
    flip = pred0 != g
    total_flips = int(flip.sum())

    # per-edge target delta_ij = b_i - b_j = -median(m over edge-{i,j} flips), m = phi_i - phi_j
    edges: list[tuple[int, int, float, float]] = []  # (i, j, delta_ij, weight = flip count)
    for i in range(k):
        gi, pi_ = (g == i), (pred0 == i)
        for j in range(i + 1, k):
            sel = flip & ((gi & (pred0 == j)) | ((g == j) & pi_))
            cnt = int(sel.sum())
            if cnt < int(edge_min_flips):
                continue
            m = z[sel, i] - z[sel, j]
            edges.append((i, j, -float(np.median(m)), float(cnt)))

    b = np.zeros(k, dtype=np.float64)
    if edges:
        lap = np.zeros((k, k), dtype=np.float64)
        rhs = np.zeros(k, dtype=np.float64)
        for i, j, d, w in edges:
            lap[i, i] += w
            lap[j, j] += w
            lap[i, j] -= w
            lap[j, i] -= w
            rhs[i] += w * d
            rhs[j] -= w * d
        b = np.linalg.pinv(lap, rcond=1e-10) @ rhs
        b = b - b.mean()                  # zero-sum gauge (pinv already min-norm; explicit for safety)
    info = {
        "converged": 1.0,                 # closed-form: always solves (b==0 when there are no flips)
        "iters": 1.0,
        "max_mass_err": float("nan"),     # mass is NOT the objective (Hamming median, not OT mass)
        "n_edges_used": float(len(edges)),
        "total_flips": float(total_flips),
        "pred_is_realized": pred_is_realized,
    }
    return b.astype(np.float64), info


# ---------------------------------------------------------------------------
# The SELECTABLE head-offset solver (#288 + #386) — the canonical dispatcher the trainer
# / probe / export path call to pick the per-class offset MECHANISM. Every arm does REAL
# work on REAL inputs (NO-FAKE): "menon" is the -tau*log(pi) prior heuristic (priors only),
# "ot_newton" is the damped-Newton OT area-mass solve (N-1-falsified for d_seg), "flip_weighted"
# is the SAME OT solve but targeting per-class FLIP SHARE, "flip_median" is the Hamming-optimal
# per-edge median (a distinct closed-form path). Each RAISES if its required inputs are absent —
# none silently degenerates to another (that would be a fake).
# ---------------------------------------------------------------------------
HEAD_OFFSET_SOLVERS: tuple[str, ...] = ("menon", "ot_newton", "flip_weighted", "flip_median")


def solve_head_offsets(
    mode: str,
    *,
    priors: np.ndarray | None = None,
    phi: np.ndarray | None = None,
    target_masses: np.ndarray | None = None,
    gt: np.ndarray | None = None,
    pred: np.ndarray | None = None,
    tau: float = 1.0,
    precondition: bool = False,
    precond_eps_rel: float = 1e-9,
    precond_cond_gate: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return the zero-sum per-class Laguerre offset ``b*`` (K,) for ``mode`` + an ``info`` dict.

    * ``mode == "menon"`` — the Menon (2007.07314) ``b_k = -tau*log(pi_k)`` heuristic
      (:func:`menon_logit_adjustment_offsets`). Needs ``priors`` (``(K,)`` non-negative class
      frequencies; ``target_masses`` accepted as an alias when ``priors`` is None). IGNORES the
      witness logit geometry. ``info`` = ``{solver: 0.0=menon, iters: 0, converged: 1}``.
    * ``mode == "ot_newton"`` — the damped-Newton semi-discrete OT solve
      (:func:`damped_newton_ot_offsets`). Needs BOTH ``phi`` (``(...,K)`` REAL witness SDF/logit
      field) AND ``target_masses`` (``(K,)`` GT class frequencies); solves the ``b*`` whose soft
      Laguerre cell masses EQUAL ``target_masses``, accounting for THIS witness's boundary geometry
      the log-freq heuristic ignores. ``info`` carries the solver's ``{converged, iters,
      max_mass_err, dual}`` plus ``solver: 1.0``. (N-1 MEASURED this HURTS realized d_seg — the area
      objective is wrong; kept as the falsified baseline the reformulations must beat.)
    * ``mode == "flip_weighted"`` — the SAME OT solve, but the target masses are the per-class FLIP
      SHARE (``flips_c/total_flips``, the canonical ``perclass_verdict.flip_share_by_class`` sensor)
      instead of GT area frequency. Needs BOTH ``phi`` AND ``gt`` (the flip share is DERIVED
      internally — it NEVER accepts a raw ``target_masses`` that could smuggle GT area counts back in
      and re-inherit N-1's failure). Flips are identified by the REALIZED argmax ``pred`` if given
      (the actual d_seg residual), else ``argmax(phi)`` (the phi-space proxy). ``info`` = the OT
      solver dict plus ``solver: 2.0, target_is_flip_share: 1.0, pred_is_realized``. UN-ANALYZED
      objective (crucible-3 P3 F3): OT still mass-MATCHES, so this may re-inherit cell-inflation —
      the through-R gate is the arbiter.
    * ``mode == "flip_median"`` — S1's Hamming-optimal per-edge median (:func:`flip_median_offsets`).
      Needs BOTH ``phi`` AND ``gt``; a DISTINCT closed-form path, NOT the OT target-mass machinery.
      ``info`` = the median-solver dict plus ``solver: 3.0``.

    All offsets fold BYTE-FREE into ``out_sdf.bias`` via :func:`apply_offset_to_sdf_bias`. NO-FAKE:
    every mode RAISES when its required inputs are absent — none is quietly replaced by another
    (``ot_newton`` never degenerates to the Menon prior; ``flip_weighted``/``flip_median`` never
    silently area-match). Use this ONE entry point everywhere a head-offset source is selected so the
    DSL flag, the trainer, the probe, and the export path stay consistent.
    """
    m = str(mode)
    if m == "menon":
        p = priors if priors is not None else target_masses
        if p is None:
            raise LaguerreLogitOffsetError("solve_head_offsets(mode='menon') requires priors (or target_masses)")
        b = menon_logit_adjustment_offsets(p, tau=tau)
        return b, {"solver": 0.0, "iters": 0.0, "converged": 1.0, "max_mass_err": 0.0}
    if m == "ot_newton":
        if phi is None or target_masses is None:
            raise LaguerreLogitOffsetError(
                "solve_head_offsets(mode='ot_newton') requires BOTH phi (witness logit field) AND "
                "target_masses (GT class frequencies) — it NEVER silently falls back to the Menon "
                "prior (that would be a fake 'ot_newton' ignoring the geometry it claims to solve)."
            )
        b, info = damped_newton_ot_offsets(
            phi, target_masses, tau=tau, precondition=precondition,
            precond_eps_rel=precond_eps_rel, precond_cond_gate=precond_cond_gate)
        out = {"solver": 1.0}
        out.update({k: float(v) for k, v in info.items()})
        return b, out
    if m == "flip_weighted":
        if phi is None or gt is None:
            raise LaguerreLogitOffsetError(
                "solve_head_offsets(mode='flip_weighted') requires BOTH phi AND gt — the target "
                "masses are the per-class FLIP SHARE derived from argmax(phi) vs gt, NEVER GT area "
                "frequency (passing area counts would re-inherit N-1's area-match failure)."
            )
        zz = np.asarray(phi, dtype=np.float64)
        zz = zz.reshape(-1, zz.shape[-1])
        # flips identified by the REALIZED argmax (pred) if given, else the phi-space argmax proxy.
        pred_lab = np.asarray(pred).reshape(-1) if pred is not None else np.argmax(zz, axis=1)
        fshare = _flip_share_by_class(pred_lab, gt, zz.shape[-1])
        b, info = damped_newton_ot_offsets(
            zz, fshare, tau=tau, precondition=precondition,
            precond_eps_rel=precond_eps_rel, precond_cond_gate=precond_cond_gate)
        out = {"solver": 2.0, "target_is_flip_share": 1.0,
               "pred_is_realized": 1.0 if pred is not None else 0.0}
        out.update({k: float(v) for k, v in info.items()})
        return b, out
    if m == "flip_median":
        if phi is None or gt is None:
            raise LaguerreLogitOffsetError(
                "solve_head_offsets(mode='flip_median') requires BOTH phi AND gt — the per-edge "
                "flip-weighted median needs the witness logits and the GT argmax; it is NOT "
                "expressible through the OT target-mass machinery (crucible-3 P3 F3)."
            )
        b, info = flip_median_offsets(phi, gt, pred=pred)
        out = {"solver": 3.0}
        out.update({k: float(v) for k, v in info.items()})
        return b, out
    raise LaguerreLogitOffsetError(
        f"unknown head-offset solver mode {mode!r}; expected one of {HEAD_OFFSET_SOLVERS}"
    )


# ---------------------------------------------------------------------------
# Facet 1a — fixed simplex Equiangular Tight Frame (ETF) head
# ---------------------------------------------------------------------------
def simplex_etf(num_classes: int, dim: int, *, scale: float = 1.0, seed: int = _ETF_SEED) -> np.ndarray:
    """Fixed simplex-ETF classifier weight ``(K, dim)`` (neural-collapse optimal).

    Rows ``w_k`` satisfy ``||w_k|| == scale`` and ``<w_i, w_j>/scale**2 == -1/(K-1)``
    for ``i != j`` (the maximally-equiangular configuration).  Construction (Yang
    et al. 2022): ``M = sqrt(K/(K-1)) * U @ (I_K - 1/K 1 1^T)`` where ``U`` is
    ``(dim, K)`` with orthonormal columns (``U^T U = I_K``), then ``w_k = M[:, k]``.
    ``U`` comes from a QR of a FIXED-seed Gaussian so the head is deterministically
    regenerable at inflate (FREE — not counted in the archive).  Requires
    ``dim >= K``.
    """
    K, d = int(num_classes), int(dim)
    if K < 2:
        raise LaguerreLogitOffsetError("num_classes must be >= 2")
    if d < K:
        raise LaguerreLogitOffsetError(f"dim ({d}) must be >= num_classes ({K}) for an ETF")
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((d, K))
    U, _ = np.linalg.qr(g)  # (d, K), orthonormal columns
    centering = np.eye(K) - np.full((K, K), 1.0 / K)
    M = np.sqrt(K / (K - 1.0)) * (U @ centering)  # (d, K)
    W = (M.T * float(scale)).astype(np.float64)  # (K, d), rows are the class prototypes
    return W


def etf_gram_offdiag(weight: np.ndarray) -> float:
    """Mean off-diagonal cosine of a (K, d) head — should be ``-1/(K-1)`` for an ETF."""
    W = np.asarray(weight, dtype=np.float64)
    n = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
    G = n @ n.T
    K = W.shape[0]
    off = G[~np.eye(K, dtype=bool)]
    return float(off.mean())


# ---------------------------------------------------------------------------
# Facet 1b — additive-margin softmax (CosFace / AM-Softmax)
# ---------------------------------------------------------------------------
def additive_margin_logits(logits: np.ndarray, target: np.ndarray, margin: float) -> np.ndarray:
    """Subtract ``margin`` from the TARGET-class logit (training-time AM-Softmax).

    ``logits``: ``(..., K)``; ``target``: ``(...)`` int labels.  Returns a copy with
    ``out[..., y] = logits[..., y] - margin`` for each pixel's target ``y`` (all
    other logits unchanged).  Feeding this to softmax/CE forces a margin ``m`` of
    inter-class separation, so the DECODED (margin=0) boundary sits ``m`` inside the
    correct cell -> survives the ``R`` round-trip.  0 archive bytes (loss only).
    """
    z = np.asarray(logits, dtype=np.float64).copy()
    t = np.asarray(target, dtype=np.int64)
    if t.shape != z.shape[:-1]:
        raise LaguerreLogitOffsetError(
            f"target shape {t.shape} != logits[...,:-1] {z.shape[:-1]}"
        )
    if (t < 0).any() or (t >= z.shape[-1]).any():
        raise LaguerreLogitOffsetError("target labels out of range [0, K)")
    np.put_along_axis(z, t[..., None], np.take_along_axis(z, t[..., None], axis=-1) - float(margin), axis=-1)
    return z


# ---------------------------------------------------------------------------
# THE FIRE-FIRST ENGINE — per-class Laguerre offset sweep on a real phi field
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OffsetSweepResult:
    """Result of a per-class Laguerre-offset sweep (task-space argmax disagreement).

    ``best_offsets`` folds into ``out_sdf.bias`` (byte-free).  ``baseline_d_seg`` is
    the un-offset task-space d_seg; ``best_d_seg`` the swept minimum; ``delta`` =
    best - baseline (NEGATIVE == improvement).  ``per_class_disagree_*`` break the
    d_seg down by GT class (the Lane long tail is class 1).  Task-space proxy:
    NON-PROMOTABLE until re-confirmed through R by ``cpu_verdict_d_seg_batch``.
    """

    best_offsets: np.ndarray
    baseline_d_seg: float
    best_d_seg: float
    delta: float
    focus_classes: tuple[int, ...]
    offset_grid: tuple[float, ...]
    per_class_disagree_baseline: dict[int, float]
    per_class_disagree_best: dict[int, float]
    table: list[dict] = field(default_factory=list)


def per_class_disagreement(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> dict[int, float]:
    """Per-GT-class argmax disagreement fraction ``{c: wrong_c / gt_px_c}``."""
    pred = np.asarray(pred).reshape(-1)
    gt = np.asarray(gt).reshape(-1)
    out: dict[int, float] = {}
    wrong = pred != gt
    for c in range(num_classes):
        m = gt == c
        n = int(m.sum())
        out[c] = float(np.count_nonzero(wrong & m)) / n if n else 0.0
    return out


def laguerre_offset_sweep(
    phi_field: np.ndarray,
    gt_argmax: np.ndarray,
    *,
    focus_classes: tuple[int, ...] = (1, 3),
    offset_grid: tuple[float, ...] = (0.0, 0.1, 0.2, 0.35, 0.5),
    base_offsets: np.ndarray | None = None,
    num_classes: int | None = None,
) -> OffsetSweepResult:
    """Sweep per-class ADDITIVE offsets over ``focus_classes`` -> best Laguerre weights.

    ``phi_field``: ``(..., K)`` REAL witness SDF logits (any leading shape).
    ``gt_argmax``: matching ``(...)`` GT SegNet argmax labels (the target partition).
    ``focus_classes``: the classes whose offsets are swept jointly over the full
    Cartesian ``offset_grid`` (default Lane=1, Movable=3 — the erasure tail);
    non-focus classes stay at ``base_offsets`` (default 0).  The objective is the
    task-space d_seg ``mean(argmax(phi+b) != gt_argmax)``; returns the argmin.

    Cost is ``len(offset_grid)**len(focus_classes)`` pure-numpy argmax passes (no
    render, no scorer) — a $0 ranking probe.  The winner is a Laguerre weight vector
    to fold into ``out_sdf.bias`` (facet 3) and to seed the trainable bias.
    """
    phi = np.asarray(phi_field, dtype=np.float64)
    K = int(num_classes) if num_classes is not None else phi.shape[-1]
    if phi.shape[-1] != K:
        raise LaguerreLogitOffsetError(f"phi last dim {phi.shape[-1]} != K {K}")
    gt = np.asarray(gt_argmax).reshape(phi.shape[:-1]).astype(np.int64)
    if base_offsets is None:
        base = np.zeros(K, dtype=np.float64)
    else:
        base = np.asarray(base_offsets, dtype=np.float64).reshape(-1).copy()
        if base.shape[0] != K:
            raise LaguerreLogitOffsetError("base_offsets K mismatch")
    for c in focus_classes:
        if not 0 <= int(c) < K:
            raise LaguerreLogitOffsetError(f"focus class {c} out of [0,K)")

    flat_phi = phi.reshape(-1, K)
    flat_gt = gt.reshape(-1)

    def dseg(b: np.ndarray) -> tuple[float, np.ndarray]:
        pred = np.argmax(flat_phi + b, axis=-1)
        return float(np.count_nonzero(pred != flat_gt)) / flat_gt.size, pred

    baseline_d, base_pred = dseg(base)
    per_class_base = per_class_disagreement(base_pred, flat_gt, K)

    # Cartesian product over the focus classes.
    import itertools

    best_b = base.copy()
    best_d = baseline_d
    best_pred = base_pred
    table: list[dict] = []
    for combo in itertools.product(offset_grid, repeat=len(focus_classes)):
        b = base.copy()
        for c, val in zip(focus_classes, combo):
            b[c] = base[c] + float(val)
        d, pred = dseg(b)
        table.append({"offsets": {int(c): float(b[c]) for c in focus_classes}, "d_seg": d})
        if d < best_d - 1e-15:
            best_d, best_b, best_pred = d, b.copy(), pred

    per_class_best = per_class_disagreement(best_pred, flat_gt, K)
    return OffsetSweepResult(
        best_offsets=best_b.astype(np.float64),
        baseline_d_seg=baseline_d,
        best_d_seg=best_d,
        delta=best_d - baseline_d,
        focus_classes=tuple(int(c) for c in focus_classes),
        offset_grid=tuple(float(v) for v in offset_grid),
        per_class_disagree_baseline=per_class_base,
        per_class_disagree_best=per_class_best,
        table=table,
    )
