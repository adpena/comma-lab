# ADMM_WAIVED:B4-reviewed historical/planning naming; docstrings or delegated coordinator code clarify whether this is Lagrangian, bridge, or actual iterative ADMM.
"""Contest objective coupling for the comma video compression challenge.

The contest scorer (`upstream/evaluate.py`) computes a non-linear
score from three components:

.. math::

    S = 100 \\cdot d_{\\text{seg}} + \\sqrt{10 \\cdot d_{\\text{pose}}}
        + \\frac{25 \\cdot B}{37{,}545{,}489}

This module provides the **canonical contest-objective coupling**: the
formula itself, its marginals, and the score-decomposition used by the
existing rate-distortion machinery. Everything else - Boyd's ADMM,
KKT residuals, Pareto frontier mapping, meta-Lagrangian search - is
already canonical in:

- :mod:`tac.joint_admm_coordinator` - Boyd 2011 section 3.4 adaptive-rho ADMM,
  per-stream ``StreamProximalCodec`` Protocol, KKT waterline residual.
- :mod:`tac.shannon_h2_loss` - differentiable H0/H2 surrogates with
  per-tensor gradient flow (the rate proxy R(theta) under the empirical
  brotli about 1.015x H0 calibration on PR106).
- :mod:`tools.apogee_intN_pareto` - Pareto frontier mapping for
  apogee_intN; the canonical mapper extended for any rate-bits axis.
- :mod:`tools.meta_lagrangian_search_cli` - meta-Lagrangian outer
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

- ``dS/dd_seg  = 100`` (constant)
- ``dS/dd_pose = sqrt(10) / (2 * sqrt(d_pose))`` (diverges as ``d_pose`` approaches zero)
- ``dS/dB      = 25 / 37,545,489`` per byte

At the old 1.x score band (``d_pose`` around 0.18), pose marginal is about 12
and SegNet dominates 8x. At the PR103-on-PR106 anchor (``d_pose`` 0.0000336),
pose marginal is about 273 and pose dominates 2.7x. The Lagrangian multipliers
must be operating-point-aware: a Joint-ADMM run that ignores this flip
will mis-allocate byte budget at the new frontier.

This module's :func:`contest_score_marginals` is the canonical
operating-point-aware sensitivity oracle that the Joint-ADMM
coordinator queries each iteration to update its per-stream
``dScore/dByte`` marginals.

# Empirical anchors (2026-05-07)

- PR103-on-PR106 standalone: 185,578 B, strict formula score
  **0.2089810755823297** from the reported component distances and exact
  charged archive bytes. The upstream-report-reconstructed score is
  ``0.20898105277982337`` because the report exposes the compression rate
  rounded to 8 decimal places. Both are A++ exact CUDA T4 custody fields; use
  the strict score for clean-checkout anchor comparisons.
- Path B Shannon analysis on PR106: brotli at 1.015x H0, 1.91x H2
  compression headroom for context-aware coders.
- Delta-epsilon-zeta training target gap (current H0-H2 gap): 78,580 B; top 5 ``blocks.*``
  tensors hold 78.5%.

# Strict-scorer-rule

Pure CPU + torch. NO scorer load. The contest formula is mathematics,
not measurement; ``contest_score`` returns the formula's output given
caller-supplied component values. The actual ``d_seg`` and ``d_pose``
values come from contest-CUDA replay via ``upstream/evaluate.py``;
this module never invokes them.

# References

- Boyd, Parikh, Chu, Peleato, Eckstein 2011 section 3.4 - adaptive-rho ADMM
- Path B per-tensor Shannon: ``tools/per_tensor_shannon_analysis.py``
- Delta-epsilon-zeta training targets: ``tools/build_deltaepszeta_training_targets.py``
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
    """Return dS/d(component) at the given operating point.

    The pose marginal diverges as ``d_pose`` approaches zero, which drives the
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
    spent. On PR103-on-PR106 (185,578 B, 0.20898 [A++ exact CUDA T4])::

        seg_term  about 0.067   (32% of total)
        pose_term about 0.018   (9%)
        rate_term about 0.124   (59%)

    Rate dominates at this operating point, which is why the council
    prescribed shipping rate-only candidates (PR103-on-PR106) AND
    queueing delta-epsilon-zeta training to attack the seg + pose terms simultaneously.
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

    Solving ``dS/dd_pose = dS/dd_seg`` for ``d_pose``::

        sqrt(10) / (2 * sqrt(d_pose)) = 100
        d_pose = 10 / (4 * 100**2) = 2.5e-4

    Below this threshold, the pose marginal exceeds the seg marginal
    (pose-dominated regime); above, seg dominates. PR106 frontier
    operates at ``d_pose`` 3.36e-5, about 7x below the threshold.
    """
    return CONTEST_POSE_WEIGHT / (4.0 * CONTEST_SEG_WEIGHT**2)


# ---------------------------------------------------------------------------
# Joint-ADMM integration adapter (bug-hunter v3 integration seam, 2026-05-07)
# ---------------------------------------------------------------------------

def joint_admm_marginal_for_stream(
    stream_role: str,
    *,
    seg_distortion: float,
    pose_distortion: float,
    archive_bytes: float,
) -> float:
    """Return the per-stream ``score_per_byte_marginal`` for Joint-ADMM.

    Bug-hunter v3 integration-seam fix 2026-05-07: this module's docstring
    documents :func:`contest_score_marginals` as the canonical sensitivity
    oracle the Joint-ADMM coordinator queries each iteration. The actual
    coordinator (:mod:`tac.joint_admm_coordinator`) consumes a *per-stream*
    ``score_per_byte_marginal`` (via ``StreamSource``) but does not import
    this module — there was no helper to convert the contest-formula
    marginals into per-stream ``dScore/dByte`` values, so callers ad-hocced
    constants (e.g. ``Op_GammaJointADMM`` defaults to ``1e-6``).

    This helper is the canonical bridge. ``stream_role`` is one of:

        - ``"weights"`` / ``"renderer"``: a stream whose byte-spend reduces
          archive_bytes; marginal = ``-dS/dB`` (the rate gradient w.r.t.
          this stream is the global rate gradient flipped: spending more
          bytes increases score).
        - ``"seg_correction"``: a stream whose byte-spend reduces
          ``d_seg`` per byte at some empirical efficiency. The default
          assumption is that one extra byte buys
          ``d_seg / archive_bytes`` reduction in seg distortion (uniform
          model); the marginal is then
          ``dS/dseg * (d_seg / archive_bytes)``. Operators with empirical
          ``d(d_seg)/d(B)`` measurements should pass them via
          :func:`joint_admm_marginal_from_empirical`.
        - ``"pose_correction"``: same model for pose. Note the pose
          marginal diverges as ``pose_distortion -> 0`` (importance-flip
          regime), so this stream becomes very hungry at PR106-band
          operating points.

    The returned value is positive (sign convention from
    :class:`tac.joint_admm_coordinator.ProximalStepResult`: positive
    means "spending more bytes lowers score"). For the ``"weights"``
    role the rate-marginal IS positive — at the operating point we
    operate, lowering archive size lowers score, so the score-per-byte
    marginal of weight-stream bytes is positive (spending fewer bytes
    on weights lowers score by ``dS/dB`` per byte saved).

    This is a CPU-only mathematical helper. Strict-scorer-rule: no
    scorer load, no measurement, just the formula's derivative.
    """
    marginals = contest_score_marginals(
        seg_distortion=seg_distortion,
        pose_distortion=pose_distortion,
        archive_bytes=archive_bytes,
    )
    if stream_role in ("weights", "renderer", "rate"):
        return float(marginals["dS_dbytes"])
    if stream_role == "seg_correction":
        # Uniform-efficiency assumption: 1 byte buys
        # (seg_distortion / archive_bytes) reduction. Caller can override
        # with empirical measurement via joint_admm_marginal_from_empirical.
        eff = float(seg_distortion) / max(float(archive_bytes), 1.0)
        return float(marginals["dS_dseg"]) * eff
    if stream_role == "pose_correction":
        eff = float(pose_distortion) / max(float(archive_bytes), 1.0)
        return float(marginals["dS_dpose"]) * eff
    raise ValueError(
        f"unknown stream_role {stream_role!r}; expected one of "
        "{'weights','renderer','rate','seg_correction','pose_correction'}"
    )


def joint_admm_marginal_from_empirical(
    *,
    delta_score_per_byte: float,
) -> float:
    """Pass-through for caller-supplied empirical ``dScore/dByte``.

    Bug-hunter v3 integration-seam fix 2026-05-07: when an operator has a
    pre-measured per-stream ``dScore/dByte`` (e.g. from a lane-G v3 frontier
    sampling pass), this helper exists for symmetry with
    :func:`joint_admm_marginal_for_stream` so callers always go through the
    canonical adapter. The function asserts the marginal is finite and
    non-negative (sign convention of
    :class:`tac.joint_admm_coordinator.ProximalStepResult`: positive means
    "spending more bytes lowers score").
    """
    if not math.isfinite(delta_score_per_byte):
        raise ValueError(
            f"delta_score_per_byte must be finite; got {delta_score_per_byte!r}"
        )
    if delta_score_per_byte < 0.0:
        raise ValueError(
            f"delta_score_per_byte must be >= 0 by ProximalStepResult sign "
            f"convention (positive ⇒ more bytes lower score); got "
            f"{delta_score_per_byte!r}. If your measurement is "
            "negative-by-construction, flip its sign at the call site to "
            "match the coordinator's convention."
        )
    return float(delta_score_per_byte)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _ensure_scalar_tensor(x: float | torch.Tensor) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if x.ndim != 0:
            raise ValueError(f"expected scalar tensor, got shape {tuple(x.shape)}")
        return x
    return torch.tensor(float(x), dtype=torch.float64)


__all__ = [
    "CONTEST_POSE_WEIGHT",
    "CONTEST_RATE_WEIGHT",
    "CONTEST_RAW_VIDEO_BYTES",
    "CONTEST_SEG_WEIGHT",
    "contest_score",
    "contest_score_decomposition",
    "contest_score_marginals",
    "importance_flip_threshold",
    "joint_admm_marginal_for_stream",
    "joint_admm_marginal_from_empirical",
]
