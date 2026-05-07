"""Contest objective coupling for the comma video compression challenge.

The contest scorer (`upstream/evaluate.py`) computes a non-linear
score from three components:

.. math::

    S = 100 \\cdot d_{\\text{seg}} + \\sqrt{10 \\cdot d_{\\text{pose}}}
        + \\frac{25 \\cdot B}{37{,}545{,}489}

This module provides the **canonical contest-objective coupling**: the
formula itself, its marginals, and the score-decomposition used by the
existing rate-distortion machinery. Everything else — Boyd's ADMM,
KKT residuals, Pareto frontier mapping, meta-Lagrangian search — is
already canonical in:

- :mod:`tac.joint_admm_coordinator` — Boyd 2011 §3.4 adaptive-ρ ADMM,
  per-stream ``StreamProximalCodec`` Protocol, KKT waterline residual.
- :mod:`tac.shannon_h2_loss` — differentiable H₀/H₂ surrogates with
  per-tensor gradient flow (the rate proxy R(θ) under the empirical
  brotli ≈ 1.015× H₀ calibration on PR106).
- :mod:`tools.apogee_intN_pareto` — Pareto frontier mapping for
  apogee_intN; the canonical mapper extended for any rate-bits axis.
- :mod:`tools.meta_lagrangian_search_cli` — meta-Lagrangian outer
  loop over candidate atoms; consumes the per-tensor weights produced
  by :mod:`tools.build_deltaepszeta_training_targets`.

This module's **unique contribution** is the contest formula and its
marginals, plus a small adapter that exposes ``S(d_seg, d_pose, B)`` as
a stationary point criterion for the existing Joint-ADMM coordinator.
The cathedral is: this module + joint_admm_coordinator + shannon_h2_loss
+ build_deltaepszeta_training_targets + apogee_intN_pareto +
meta_lagrangian_search_cli together form the full optimization stack.

# Why the contest formula matters for the Lagrangian

The sqrt non-linearity on pose creates an **operating-point-dependent
importance flip** (CLAUDE.md "SegNet vs PoseNet importance"):

- ``∂S/∂d_seg  = 100`` (constant)
- ``∂S/∂d_pose = sqrt(10) / (2·sqrt(d_pose))`` (diverges as ``d_pose → 0``)
- ``∂S/∂B     = 25 / 37,545,489 ≈ 6.66e-7`` per byte

At the OLD 1.x score band (``d_pose ≈ 0.18``), pose marginal ≈ 12 and
SegNet dominates 8×. At the PR106 frontier (``d_pose ≈ 3.4e-5``), pose
marginal ≈ 271 and POSE dominates 2.7×. The Lagrangian multipliers
must be operating-point-aware: a Joint-ADMM run that ignores this flip
will mis-allocate byte budget at the new frontier.

This module's :func:`contest_score_marginals` is the canonical
operating-point-aware sensitivity oracle that the Joint-ADMM
coordinator queries each iteration to update its per-stream
``dScore/dByte`` marginals.

# Empirical anchors (2026-05-07)

- PR103-on-PR106 standalone: 185,578 B → score **0.20898105277982337**
  ``[contest-CUDA T4]`` (5/5 gates GREEN; new local frontier).
- Path B Shannon analysis on PR106: brotli at 1.015× H₀, 1.91× H₂
  compression headroom for context-aware coders.
- δεζ training prize (current H₀-H₂ gap): 78,580 B; top 5 ``blocks.*``
  tensors hold 78.5%.

# Strict-scorer-rule

Pure CPU + torch. NO scorer load. The contest formula is mathematics,
not measurement; ``contest_score`` returns the formula's output given
caller-supplied component values. The actual ``d_seg`` and ``d_pose``
values come from contest-CUDA replay via ``upstream/evaluate.py``;
this module never invokes them.

# References

- Boyd, Parikh, Chu, Peleato, Eckstein 2011 §3.4 — adaptive-ρ ADMM
- Path B per-tensor Shannon: ``tools/per_tensor_shannon_analysis.py``
- δεζ training targets: ``tools/build_deltaepszeta_training_targets.py``
- Council deliberation: ``.omx/research/grand_council_pr106_substrate_findings_zig_default_20260507.md``
- Frontier anchor: ``experiments/results/pr103_repack_pr106_standalone_20260507/``
"""

from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# Contest constants (frozen by upstream/evaluate.py)
# ---------------------------------------------------------------------------

CONTEST_SEG_WEIGHT: float = 100.0
"""SegNet distortion weight in the contest score formula."""

CONTEST_POSE_WEIGHT: float = 10.0
"""PoseNet distortion weight inside the sqrt non-linearity."""

CONTEST_RATE_WEIGHT: float = 25.0
"""Rate weight (numerator) in the contest score formula."""

CONTEST_RAW_VIDEO_BYTES: int = 37_545_489
"""Raw video size in bytes (denominator of rate term)."""


# ---------------------------------------------------------------------------
# Contest objective + marginals (the unique contribution of this module)
# ---------------------------------------------------------------------------

def contest_score(
    *,
    seg_distortion: float | torch.Tensor,
    pose_distortion: float | torch.Tensor,
    archive_bytes: float | torch.Tensor,
) -> torch.Tensor:
    """Compute the contest score scalar.

    .. math::

        S = 100 \\cdot d_{\\text{seg}} + \\sqrt{10 \\cdot d_{\\text{pose}}}
            + \\frac{25 \\cdot B}{37{,}545{,}489}

    Differentiable in all three inputs (sqrt is autograd-friendly with
    a small clamp-min for ``d_pose = 0``).
    """
    seg_t = _ensure_scalar_tensor(seg_distortion)
    pose_t = _ensure_scalar_tensor(pose_distortion)
    bytes_t = _ensure_scalar_tensor(archive_bytes)
    return (
        CONTEST_SEG_WEIGHT * seg_t
        + torch.sqrt(CONTEST_POSE_WEIGHT * pose_t.clamp_min(1e-30))
        + CONTEST_RATE_WEIGHT * bytes_t / CONTEST_RAW_VIDEO_BYTES
    )


def contest_score_marginals(
    *,
    seg_distortion: float,
    pose_distortion: float,
    archive_bytes: float,
) -> dict[str, float]:
    """Return ∂S/∂(component) at the given operating point.

    The pose marginal diverges as ``d_pose → 0``, which drives the
    operating-point-dependent importance flip documented in CLAUDE.md.
    Joint-ADMM consumers should query this each iteration to update
    per-stream ``dScore/dByte`` allocations.
    """
    pose_floor = max(pose_distortion, 1e-30)
    return {
        "dS_dseg": CONTEST_SEG_WEIGHT,
        "dS_dpose": math.sqrt(CONTEST_POSE_WEIGHT) / (2.0 * pose_floor**0.5),
        "dS_dbytes": CONTEST_RATE_WEIGHT / CONTEST_RAW_VIDEO_BYTES,
        "value": float(contest_score(
            seg_distortion=seg_distortion,
            pose_distortion=pose_distortion,
            archive_bytes=archive_bytes,
        )),
    }


def contest_score_decomposition(
    *,
    seg_distortion: float,
    pose_distortion: float,
    archive_bytes: float,
) -> dict[str, float]:
    """Return the per-term contribution to the contest score.

    Useful for forensic analysis of where the score budget is being
    spent. On PR103-on-PR106 (185,578 B → 0.20898 [contest-CUDA T4])::

        seg_term  ≈ 0.067   (32% of total)
        pose_term ≈ 0.018   (9%)
        rate_term ≈ 0.124   (59%)

    Rate dominates at this operating point — which is why the council
    prescribed shipping rate-only candidates (PR103-on-PR106) AND
    queueing δεζ training to attack the seg + pose terms simultaneously.
    """
    seg_term = CONTEST_SEG_WEIGHT * float(seg_distortion)
    pose_term = math.sqrt(CONTEST_POSE_WEIGHT * max(float(pose_distortion), 1e-30))
    rate_term = CONTEST_RATE_WEIGHT * float(archive_bytes) / CONTEST_RAW_VIDEO_BYTES
    total = seg_term + pose_term + rate_term
    return {
        "seg_term": seg_term,
        "pose_term": pose_term,
        "rate_term": rate_term,
        "total": total,
        "seg_share": seg_term / total if total > 0 else 0.0,
        "pose_share": pose_term / total if total > 0 else 0.0,
        "rate_share": rate_term / total if total > 0 else 0.0,
    }


def importance_flip_threshold() -> float:
    """Pose-distortion value at which the pose marginal equals the seg marginal.

    Solving ``∂S/∂d_pose = ∂S/∂d_seg`` for ``d_pose``::

        sqrt(10) / (2·sqrt(d_pose)) = 100
        d_pose = 10 / (4·100²) = 2.5e-4

    Below this threshold, the pose marginal exceeds the seg marginal
    (pose-dominated regime); above, seg dominates. PR106 frontier
    operates at ``d_pose ≈ 3.4e-5``, which is ~7× below the threshold.
    """
    return CONTEST_POSE_WEIGHT / (4.0 * CONTEST_SEG_WEIGHT**2)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _ensure_scalar_tensor(x: float | torch.Tensor) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if x.ndim != 0:
            raise ValueError(f"expected scalar tensor, got shape {tuple(x.shape)}")
        return x
    return torch.tensor(float(x))


__all__ = [
    "CONTEST_POSE_WEIGHT",
    "CONTEST_RATE_WEIGHT",
    "CONTEST_RAW_VIDEO_BYTES",
    "CONTEST_SEG_WEIGHT",
    "contest_score",
    "contest_score_decomposition",
    "contest_score_marginals",
    "importance_flip_threshold",
]
