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


def damped_newton_ot_offsets(
    phi: np.ndarray, target_masses: np.ndarray, *, tau: float = 1.0,
    max_iter: int = 64, tol: float = 1e-10, rcond: float = 1e-10,
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
    Returns ``(b*, info)`` with ``info`` = {converged, iters, max_mass_err, dual}.
    Byte-free: fold ``b*`` into the bias via :func:`apply_offset_to_sdf_bias`.
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

    info = {"converged": 0.0, "iters": 0.0, "max_mass_err": 1.0, "dual": _dual(b)}
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
        step = taus * (np.linalg.pinv(cov, rcond=rcond) @ g)       # Newton direction (dual ascent)
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
