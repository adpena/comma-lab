# SPDX-License-Identifier: MIT
"""Lever C — per-dim pose Mahalanobis / AIL weighting + score-irrelevant-dim handling.

Design memo: ``.omx/research/sophisticated_pose_treatment_design_20260616T222900Z.md``.

THE PRINCIPLE (the optimal-teacher pose design, made into a per-dim weight on the driver's pose loss).
The contest ``d_pose`` is the MSE on the FIRST 6 of the 12 PoseNet pose dims (``upstream/modules.py`` +
the ``[:6]`` slice the driver already applies). The driver's pose loss is currently a UNIFORM
``MSE(pose_pred6, target6)`` — every scored dim weighted equally. The score-optimal teacher (memory
``feedback_optimal_teacher_and_sensitivity_tools_landed_20260531.md``: per-dim Mahalanobis reweight +
Saputra-2019 AIL) weights each scored dim by its inverse per-dim variance (so high-variance / hard dims
get more pull) and, optionally, hard-zeros a dim whose Jacobian row carries negligible score sensitivity.

THE WEIGHT (NO-FAKE, byte-identical default). Lever C threads a length-6 weight vector ``w`` into the pose
loss::

    pose_mse_weighted = mean_pairs( sum_k w_k · (pred_k − tgt_k)^2 ) / mean(w)      # renormalised

``w`` is RENORMALISED so the default uniform ``w = (1,1,1,1,1,1)`` gives EXACTLY the current
``mean_k (pred_k − tgt_k)^2`` (byte-identical). ``None`` (the default) skips the weighting entirely (the
driver's existing ``F.mse_loss`` path is unchanged — not even a multiply). The renormalisation
``/ mean(w)`` keeps the OVERALL pose-loss magnitude (and hence ``pose_weight``'s calibration) stable as the
weights tilt the per-dim allocation — only the BALANCE between dims changes, not the total scale.

THE WEIGHTS ARE MEASURED, NOT HAND-SET. :func:`measure_pose_dim_weights_from_targets` derives ``w`` from the
REAL per-dim target variance over the basin's pose targets (the inverse-variance Mahalanobis form), and
:func:`zero_low_sensitivity_dims` can hard-zero dims whose measured Jacobian row energy is below a tolerance
(reusing the #80 spectrum / #61 saliency machinery upstream). The default is uniform; any non-uniform weight
must come from a measurement artifact (the basin variance / Jacobian), per the design's
UNCLEAR_NEEDS_EMPIRICAL classification of "6 dims equally sensitive".

DEFAULT-OFF / BYTE-IDENTICAL. ``cfg.pose_dim_weights is None`` → the driver's pose path is untouched
(no multiply, no renorm). A length-6 tuple routes the squared error through :func:`weighted_pose_mse`.

NO-FAKE. :func:`weighted_pose_mse` is the exact weighted MSE (a test asserts uniform weights == the plain
MSE to float tolerance, and non-uniform weights tilt the loss in the measured direction). The measurement
helpers read REAL targets / REAL Jacobian rows; they do not fabricate a weight vector.
"""

from __future__ import annotations

__all__ = [
    "N_SCORED_POSE_DIMS",
    "measure_pose_dim_weights_from_targets",
    "normalise_pose_dim_weights",
    "weighted_pose_mse",
    "zero_low_sensitivity_dims",
]

N_SCORED_POSE_DIMS = 6  # d_pose = MSE on first 6 of 12 pose dims (upstream/modules.py)
_EPS = 1e-12


def normalise_pose_dim_weights(weights) -> tuple[float, ...]:
    """Return a length-6 weight tuple RENORMALISED to mean 1.0 (so the overall pose-loss scale is preserved;
    only the per-dim balance changes). Validates length 6 + nonneg + not-all-zero. The uniform input
    ``(1,)*6`` returns ``(1,)*6`` exactly."""
    w = [float(x) for x in weights]
    if len(w) != N_SCORED_POSE_DIMS:
        raise ValueError(f"pose dim weights must have length {N_SCORED_POSE_DIMS}, got {len(w)}")
    if any(x < 0.0 for x in w):
        raise ValueError(f"pose dim weights must be >= 0, got {w}")
    mean = sum(w) / N_SCORED_POSE_DIMS
    if mean <= _EPS:
        raise ValueError("pose dim weights cannot be all-zero (would zero the pose loss)")
    return tuple(x / mean for x in w)


def weighted_pose_mse(pred6, target6, weights=None):
    """Weighted per-dim pose MSE: ``mean_pairs( sum_k w_k·(pred−tgt)^2 ) / mean(w)``.

    ``pred6`` / ``target6``: ``(B, 6)`` torch tensors (the first-6 scored pose dims). ``weights``: a
    length-6 sequence (any positive scale; renormalised internally) or ``None`` for plain uniform MSE.

    Returns a scalar torch tensor. With ``weights=None`` OR uniform weights this equals ``F.mse_loss(pred6,
    target6)`` to float tolerance (byte-identical default). The squared error is reduced over the batch with
    ``mean`` and over the 6 dims with the renormalised weighted sum / 6 (the ``/ mean(w)`` keeps the scale)."""
    import torch

    sq = (pred6 - target6) ** 2  # (B, 6)
    if weights is None:
        return sq.mean()
    wn = normalise_pose_dim_weights(weights)  # mean 1.0
    wt = torch.as_tensor(wn, dtype=sq.dtype, device=sq.device)  # (6,)
    # sum_k w_k·sq_k / 6 == mean_k (w_k·sq_k) since mean(w)==1 → matches plain mean for uniform w.
    return (sq * wt).mean()


def measure_pose_dim_weights_from_targets(pose_targets, *, mode: str = "inv_var", floor: float = 0.1):
    """Derive the length-6 per-dim weights from the REAL basin pose targets (Mahalanobis inverse-variance).

    ``pose_targets``: ``(n_pairs, 6)`` (or wider; the first 6 are used) torch tensor / array of the basin's
    scored pose targets. ``mode``:
      * ``"inv_var"`` (default): ``w_k ∝ 1 / (var_k + eps)`` — the Mahalanobis form (high-variance dims, which
        contribute most to the MSE, get MORE pull). Renormalised to mean 1.0.
      * ``"var"``: ``w_k ∝ var_k`` (the opposite tilt; for ablation).
      * ``"uniform"``: returns ``(1,)*6`` (the byte-identical default; useful as a measured control).
    ``floor``: a lower bound on each pre-normalisation weight (as a fraction of the max) so a near-constant
    dim is not driven to ~0 (which would effectively drop a scored dim — use :func:`zero_low_sensitivity_dims`
    for an INTENTIONAL drop). Returns a length-6 tuple (mean 1.0). NO-FAKE: the variance is the EXACT
    per-dim variance of the real targets; no fabricated numbers."""
    import torch

    t = torch.as_tensor(pose_targets, dtype=torch.float64)
    if t.ndim != 2 or t.shape[1] < N_SCORED_POSE_DIMS:
        raise ValueError(f"pose_targets must be (n_pairs, >=6), got {tuple(t.shape)}")
    t6 = t[:, :N_SCORED_POSE_DIMS]
    var = t6.var(dim=0, unbiased=False)  # (6,)
    if mode == "uniform":
        return tuple([1.0] * N_SCORED_POSE_DIMS)
    if mode == "var":
        raw = var + _EPS
    elif mode == "inv_var":
        raw = 1.0 / (var + _EPS)
    else:
        raise ValueError(f"mode must be inv_var|var|uniform, got {mode!r}")
    # floor each weight at `floor`·max so no scored dim is starved to ~0 unintentionally.
    mx = float(raw.max().item())
    if mx <= _EPS:
        return tuple([1.0] * N_SCORED_POSE_DIMS)
    raw = torch.clamp(raw, min=floor * mx)
    return normalise_pose_dim_weights([float(x) for x in raw.tolist()])


def zero_low_sensitivity_dims(weights, jac_row_energy, *, rel_tol: float = 1e-3):
    """Hard-zero scored dims whose measured Jacobian-row ENERGY is below ``rel_tol`` × the max row energy,
    then renormalise the survivors to mean 1.0 over the SURVIVING dims (the zeroed dims drop out of d_pose).

    ``weights``: a length-6 weight sequence. ``jac_row_energy``: length-6 per-dim energy ``‖J_k‖^2`` of the
    6×N PoseNet Jacobian rows (from ``measure_pose_subspace_spectrum`` upstream — the row 2-norms squared, or
    the singular contributions). A dim with ~0 row energy reads ~0 pose sensitivity → its squared error
    cannot move d_pose to first order, so it is safe to drop (frees the pose loss to concentrate on the dims
    that matter). Returns ``(weights_tuple, zeroed_indices)``. If NO dim is below tolerance the input is
    returned (renormalised) with an empty zeroed list — byte-identical to keeping all 6. NO-FAKE: the drop
    decision is on MEASURED Jacobian energy; a test proves a low-energy dim is dropped + the survivors
    renormalise to mean 1.0."""
    e = [float(x) for x in jac_row_energy]
    w = [float(x) for x in weights]
    if len(e) != N_SCORED_POSE_DIMS or len(w) != N_SCORED_POSE_DIMS:
        raise ValueError(
            f"jac_row_energy and weights must be length {N_SCORED_POSE_DIMS}, got {len(e)}, {len(w)}"
        )
    mx = max(e) if e else 0.0
    zeroed = [k for k in range(N_SCORED_POSE_DIMS) if mx > _EPS and e[k] < rel_tol * mx]
    if not zeroed:
        return normalise_pose_dim_weights(w), []
    survivors = [w[k] if k not in zeroed else 0.0 for k in range(N_SCORED_POSE_DIMS)]
    n_surv = N_SCORED_POSE_DIMS - len(zeroed)
    if n_surv == 0:
        raise ValueError("all 6 pose dims below sensitivity tolerance — refusing to zero the entire loss")
    mean_surv = sum(survivors) / n_surv
    if mean_surv <= _EPS:
        # survivors all-zero weight: fall back to uniform over survivors.
        survivors = [1.0 if k not in zeroed else 0.0 for k in range(N_SCORED_POSE_DIMS)]
        mean_surv = sum(survivors) / n_surv
    return tuple(x / mean_surv for x in survivors), zeroed
