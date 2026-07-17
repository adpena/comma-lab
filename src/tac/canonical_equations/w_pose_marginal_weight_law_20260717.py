# SPDX-License-Identifier: MIT
"""w_pose(t) = 5/sqrt(10 * d_pose(t)) — the score's OWN pose marginal as the pose-finish weight.

SPEC_v10 §13.3 (arm B, 2026-07-17). The contest score is
``S = 100*d_seg + sqrt(10*d_pose) + 25*bytes/37_545_489``; the pose term's exact derivative is

    dS/dd_pose = d/dd_pose sqrt(10*d_pose) = 10 / (2*sqrt(10*d_pose)) = 5 / sqrt(10*d_pose).

Setting the pose loss weight to this marginal makes ONE unit of pose-loss descent worth exactly
its score value at the CURRENT operating point (the same exchange rate the score itself applies),
instead of the static ``--w-pose 1.0`` constant (a bare constant on the value-provenance ladder).

THE CLAMP (DERIVED, not tuned): the marginal DIVERGES as ``d_pose -> 0``. The seg marginal is the
CONSTANT ``dS/dd_seg = 100``; the two marginals cross at

    5 / sqrt(10 * d_c) = 100  =>  d_c = 2.5e-4  (the CLAUDE.md operating-point crossover),

so the clamp is ``w_max = 100.0`` — the pose weight is never allowed to exceed the score's own
seg exchange rate. Beyond the crossover the score still prefers pose descent marginally, but a
loss weight above 100 would let the pose term outbid the seg gradient budget beyond any exchange
rate the score expresses between its two distortion terms; capping at the seg marginal keeps the
joint descent inside the score's own trade surface (and bounds the known deep-unroll coefficient
blowup — sister of the trainer's ``--pose-grad-coeff-max`` guard, same divergence).

Consumption point: the pose-finish stage ONLY (the trainer's ``_w_pose_now`` holder), updated at
VERDICT cadence when a measured ``d_pose`` lands — piecewise-constant between measurements, never
per-step (SPEC_v75 §8 operating contract: loss weights change at stage/measurement boundaries).

Producers: ``experiments/train_levelset_witness_realized_through_R_mlx.py`` (the
``--w-pose-marginal-law`` lever). Consumers: ``tac.witness_dsl.curriculum_dsl.
PoseMarginalWeightLaw`` (LawRef custody) + the pose-finish engage block.
"""
from __future__ import annotations

import math

EQUATION_ID = "w_pose_marginal_weight_law_v1"

#: dS/dd_seg — the score's seg marginal (exact constant from the score definition).
SEG_MARGINAL: float = 100.0

#: the marginal crossover: 5/sqrt(10*d_c) == SEG_MARGINAL  =>  d_c = 25/(10*SEG_MARGINAL^2).
D_POSE_CROSSOVER: float = 25.0 / (10.0 * SEG_MARGINAL * SEG_MARGINAL)   # == 2.5e-4 exactly


def w_pose_marginal(d_pose: float) -> float:
    """The exact pose marginal ``5/sqrt(10*d_pose)`` (unclamped). Raises on non-positive /
    non-finite input — a measured d_pose is always > 0; the caller owns the no-measurement
    fallback."""
    dp = float(d_pose)
    if not math.isfinite(dp) or dp <= 0.0:
        raise ValueError(f"w_pose_marginal: d_pose must be finite and > 0, got {d_pose!r}")
    return 5.0 / math.sqrt(10.0 * dp)


def clamp_from_crossover(seg_marginal: float = SEG_MARGINAL) -> float:
    """The DERIVED clamp = the seg marginal (the value of the pose marginal AT the crossover
    d_c = 25/(10*seg_marginal^2)). With the score's seg coefficient 100 this is exactly 100.0."""
    sm = float(seg_marginal)
    if not (sm > 0.0):
        raise ValueError(f"clamp_from_crossover: seg_marginal must be > 0, got {seg_marginal!r}")
    return sm


def w_pose_law(d_pose: float, *, clamp: float | None = None) -> float:
    """The shipped law: ``min(clamp, 5/sqrt(10*d_pose))`` with the derived crossover clamp by
    default. This is what the trainer's ``--w-pose-marginal-law`` lever computes at each verdict
    boundary while the pose-finish is engaged."""
    c = clamp_from_crossover() if clamp is None else float(clamp)
    if not (c > 0.0):
        raise ValueError(f"w_pose_law: clamp must be > 0, got {clamp!r}")
    return min(c, w_pose_marginal(d_pose))


def verify_crossover_identity() -> bool:
    """Self-check: the marginal AT the crossover equals the clamp (the derivation's algebra)."""
    return abs(w_pose_marginal(D_POSE_CROSSOVER) - SEG_MARGINAL) < 1e-9


__all__ = [
    "EQUATION_ID", "SEG_MARGINAL", "D_POSE_CROSSOVER",
    "w_pose_marginal", "clamp_from_crossover", "w_pose_law", "verify_crossover_identity",
]
