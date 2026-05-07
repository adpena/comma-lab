"""Second-order Volterra-style stacking interaction model.

The ``tac.score_geometry`` module treats each axis (seg, pose, bytes)
independently — exact for first-order analysis. But when stacking
multiple lanes that touch the same axis or different axes, the
**marginal value of the second lane depends on the first**. This is
the empirical "stacking gain ≠ sum of individual gains" finding.

This module models that interaction at second order — the leading
correction term in a Volterra expansion of the score around a focal
operating point. For the contest objective ``S = 100*d_seg +
sqrt(10*d_pose) + 25*B/N``, the only non-linearity is the pose sqrt,
so the only second-order interaction worth modeling analytically is:

  - **pose × pose** (concavity correction): two pose-improvement lanes
    stacked yield LESS than the sum of their individual gains because
    pose's marginal flattens as d_pose decreases. This is the key
    "stacking saturation" effect.

For mixed-axis stacking (seg+pose, pose+bytes, etc.), the linearity of
seg and bytes contributions means the only correction is from each
axis's own internal nonlinearity. So:

  ΔS(stack) = ΔS(lane_a) + ΔS(lane_b) + correction_pose_pose

where correction is the second-order Taylor remainder for the pose sqrt.

**Counter-intuitive: pose-pose stacking is SUPER-ADDITIVE.** Because we
move TOWARD the sqrt singularity (d_pose → 0), and the marginal grows
unboundedly in that direction, the second pose lane operates at a
steeper slope than the first. Stacked gain > sum of individuals. This
is the "compounding-toward-Shannon-floor" effect — exactly opposite of
the intuition that "concavity should saturate." The concavity is real;
the operational direction (decreasing d_pose) is what flips the sign.

For the empirical ratio
  R = ΔS(stack) / (ΔS(lane_a) + ΔS(lane_b))
this lets us **predict** R given the focal operating point and the
individual lane savings, without dispatching the stacked combination.

Closed-form for two pose lanes with deltas Δp_a, Δp_b at focal d_pose:

  S(focal) = sqrt(10 * d_pose)
  S(focal - Δp_a - Δp_b) = sqrt(10 * (d_pose - Δp_a - Δp_b))

The stacked saving Δ_stack = sqrt(10 * d_pose) - sqrt(10 * (d_pose - Δp_a - Δp_b))
The sum of individuals = (sqrt(10*d_pose) - sqrt(10*(d_pose-Δp_a)))
                      + (sqrt(10*d_pose) - sqrt(10*(d_pose-Δp_b)))

Both are computed analytically; no fitting, no curve learning.

Cross-references:
  - tac.score_geometry — the linearized base layer
  - tools/dispatch_advisor.py — first-order axis priority recommender
  - the Op3+Op1 "loses 119KB on random noise" finding — that empirical
    finding is consistent with this model PLUS the substrate-mismatch
    term (which this module documents as a separate, non-analytical
    correction).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from tac.score_geometry import (
    POSE_COEFFICIENT_INSIDE_SQRT,
    contest_score,
    score_decomposition,
)


@dataclass(frozen=True)
class StackingPrediction:
    """Predicted score after applying two lane improvements, with
    quantitative breakdown of the linear vs interaction contributions."""

    focal_d_seg: float
    focal_d_pose: float
    focal_archive_bytes: int
    focal_score: float

    delta_d_seg_a: float
    delta_d_pose_a: float
    delta_bytes_a: int

    delta_d_seg_b: float
    delta_d_pose_b: float
    delta_bytes_b: int

    score_after_a_only: float
    score_after_b_only: float
    score_after_stack: float

    individual_gain_a: float          # focal_score - score_after_a_only
    individual_gain_b: float          # focal_score - score_after_b_only
    sum_individual_gains: float       # individual_gain_a + individual_gain_b
    stacked_gain: float               # focal_score - score_after_stack

    pose_pose_interaction_correction: float
    """Closed-form second-order Volterra correction from pose-axis sqrt.
    POSITIVE when both lanes touch pose (super-additive: each pose
    improvement compounds because we move toward the sqrt singularity
    where the marginal blows up). Zero when only one lane touches pose
    or both lanes target disjoint axes."""

    nominal_stack_ratio: float
    """ΔS(stack) / Σ ΔS(individual). 1.0 means perfectly additive (linear
    axes only). >1 means super-additive (typical for pose-pose stacking
    moving toward Shannon floor). <1 would indicate substrate-mismatch
    or empirical interactions outside this analytical model."""

    notes: list[str]


def predict_stacking(
    *,
    focal_d_seg: float,
    focal_d_pose: float,
    focal_archive_bytes: int,
    delta_d_seg_a: float = 0.0,
    delta_d_pose_a: float = 0.0,
    delta_bytes_a: int = 0,
    delta_d_seg_b: float = 0.0,
    delta_d_pose_b: float = 0.0,
    delta_bytes_b: int = 0,
) -> StackingPrediction:
    """Predict the stacked-lane outcome from individual lane deltas.

    Each "lane" is a 3-axis improvement vector (delta_d_seg, delta_d_pose,
    delta_bytes). Positive deltas reduce the corresponding distortion or
    archive size. Stacking applies BOTH deltas to the focal point;
    "individual" applies just one.

    The seg and bytes axes are linear → no correction. The pose axis is
    concave → analytical correction is non-zero whenever both lanes touch
    pose with positive delta.
    """
    if (delta_d_seg_a < 0 or delta_d_pose_a < 0 or delta_bytes_a < 0
        or delta_d_seg_b < 0 or delta_d_pose_b < 0 or delta_bytes_b < 0):
        raise ValueError("lane deltas are positive improvements; negative deltas not supported")

    # Score at focal + individual + stacked points
    focal_score = contest_score(focal_d_seg, focal_d_pose, focal_archive_bytes)
    score_after_a = contest_score(
        max(focal_d_seg - delta_d_seg_a, 0.0),
        max(focal_d_pose - delta_d_pose_a, 0.0),
        max(focal_archive_bytes - delta_bytes_a, 0),
    )
    score_after_b = contest_score(
        max(focal_d_seg - delta_d_seg_b, 0.0),
        max(focal_d_pose - delta_d_pose_b, 0.0),
        max(focal_archive_bytes - delta_bytes_b, 0),
    )
    score_after_stack = contest_score(
        max(focal_d_seg - delta_d_seg_a - delta_d_seg_b, 0.0),
        max(focal_d_pose - delta_d_pose_a - delta_d_pose_b, 0.0),
        max(focal_archive_bytes - delta_bytes_a - delta_bytes_b, 0),
    )
    individual_gain_a = focal_score - score_after_a
    individual_gain_b = focal_score - score_after_b
    sum_individual_gains = individual_gain_a + individual_gain_b
    stacked_gain = focal_score - score_after_stack

    # Closed-form pose-pose interaction correction:
    # The pose contribution to ΔS is sqrt(10*p_focal) - sqrt(10*p').
    # For two stacked pose deltas, the stacked Δ minus the sum of
    # individual Δs is exactly:
    #   correction = -sqrt(10*p) + sqrt(10*(p-Δa)) + sqrt(10*(p-Δb))
    #                              - sqrt(10*(p-Δa-Δb))
    # Negative when both deltas exist (concavity).
    p = focal_d_pose
    pa = max(p - delta_d_pose_a, 0.0)
    pb = max(p - delta_d_pose_b, 0.0)
    pab = max(p - delta_d_pose_a - delta_d_pose_b, 0.0)
    pose_pose_interaction = -(
        math.sqrt(POSE_COEFFICIENT_INSIDE_SQRT * p)
        - math.sqrt(POSE_COEFFICIENT_INSIDE_SQRT * pa)
        - math.sqrt(POSE_COEFFICIENT_INSIDE_SQRT * pb)
        + math.sqrt(POSE_COEFFICIENT_INSIDE_SQRT * pab)
    )

    if sum_individual_gains > 0:
        nominal_stack_ratio = stacked_gain / sum_individual_gains
    else:
        nominal_stack_ratio = 1.0  # zero individual = no meaningful ratio

    notes: list[str] = []
    if delta_d_pose_a > 0 and delta_d_pose_b > 0:
        notes.append(
            f"Both lanes touch pose axis. Closed-form sqrt-direction "
            f"correction = {pose_pose_interaction:+.5f} "
            f"(positive = super-additive; each pose improvement "
            f"compounds toward the Shannon floor)."
        )
    if pose_pose_interaction == 0.0 and sum_individual_gains > 0:
        notes.append(
            "Lanes touch disjoint axes (or only one touches pose); "
            "stacking is purely additive."
        )
    if nominal_stack_ratio < 0.95 and pose_pose_interaction == 0.0:
        notes.append(
            "Sub-additive stacking with no analytical pose-pose interaction "
            "predicted. Possible substrate-mismatch (empirical), e.g. one "
            "lane's bytes savings depend on the other's quantization basis. "
            "Sanity-check via small empirical cross-test."
        )

    return StackingPrediction(
        focal_d_seg=focal_d_seg,
        focal_d_pose=focal_d_pose,
        focal_archive_bytes=focal_archive_bytes,
        focal_score=focal_score,
        delta_d_seg_a=delta_d_seg_a,
        delta_d_pose_a=delta_d_pose_a,
        delta_bytes_a=delta_bytes_a,
        delta_d_seg_b=delta_d_seg_b,
        delta_d_pose_b=delta_d_pose_b,
        delta_bytes_b=delta_bytes_b,
        score_after_a_only=score_after_a,
        score_after_b_only=score_after_b,
        score_after_stack=score_after_stack,
        individual_gain_a=individual_gain_a,
        individual_gain_b=individual_gain_b,
        sum_individual_gains=sum_individual_gains,
        stacked_gain=stacked_gain,
        pose_pose_interaction_correction=pose_pose_interaction,
        nominal_stack_ratio=nominal_stack_ratio,
        notes=notes,
    )


@dataclass(frozen=True)
class TripleStackingPrediction:
    """Three-lane stacking prediction with closed-form Volterra correction.

    Same shape as `StackingPrediction` for two lanes. The pose-pose-pose
    correction is the third-order Taylor remainder for the pose sqrt term
    when ALL THREE lanes touch pose. For mixed touches it reduces to the
    pairwise pose-pose corrections plus a residual.

    The four-way stack composition (cathedral phase 4 integration) becomes
    natural in this framework: encode each pair correction analytically,
    then apply the empirical substrate-mismatch correction term per lane
    pair from past sanity-check evidence.
    """

    focal_d_seg: float
    focal_d_pose: float
    focal_archive_bytes: int
    focal_score: float

    delta_d_pose_a: float
    delta_d_pose_b: float
    delta_d_pose_c: float
    delta_d_seg_a: float
    delta_d_seg_b: float
    delta_d_seg_c: float
    delta_bytes_a: int
    delta_bytes_b: int
    delta_bytes_c: int

    score_after_a_only: float
    score_after_b_only: float
    score_after_c_only: float
    score_after_ab: float
    score_after_ac: float
    score_after_bc: float
    score_after_abc: float

    individual_gain_a: float
    individual_gain_b: float
    individual_gain_c: float
    sum_individual_gains: float
    stacked_gain: float

    pairwise_pose_corrections: dict[str, float]
    triple_pose_correction: float
    """The third-order Volterra residual: stacked_gain - sum_individual_gains -
    sum of pairwise corrections. For pose-pose-pose this is the pure
    third-order term; for any mixed combination the linear axes contribute 0
    so this isolates the genuine triple interaction.
    """

    nominal_stack_ratio: float
    notes: list[str]


def predict_triple_stacking(
    *,
    focal_d_seg: float,
    focal_d_pose: float,
    focal_archive_bytes: int,
    delta_d_seg_a: float = 0.0,
    delta_d_pose_a: float = 0.0,
    delta_bytes_a: int = 0,
    delta_d_seg_b: float = 0.0,
    delta_d_pose_b: float = 0.0,
    delta_bytes_b: int = 0,
    delta_d_seg_c: float = 0.0,
    delta_d_pose_c: float = 0.0,
    delta_bytes_c: int = 0,
) -> TripleStackingPrediction:
    """Predict three-lane stacked outcome with full Volterra decomposition.

    Output exposes individual gains, pairwise stacked gains (AB / AC / BC),
    and the full triple gain (ABC). The triple_pose_correction is the
    residual after subtracting individual gains and pairwise corrections —
    pure third-order pose interaction.
    """
    for d in (
        delta_d_seg_a, delta_d_pose_a, delta_d_seg_b, delta_d_pose_b,
        delta_d_seg_c, delta_d_pose_c,
    ):
        if d < 0:
            raise ValueError("lane deltas must be positive improvements")
    if delta_bytes_a < 0 or delta_bytes_b < 0 or delta_bytes_c < 0:
        raise ValueError("byte deltas must be positive improvements")

    def _score_at(seg_drop: float, pose_drop: float, byte_drop: int) -> float:
        return contest_score(
            max(focal_d_seg - seg_drop, 0.0),
            max(focal_d_pose - pose_drop, 0.0),
            max(focal_archive_bytes - byte_drop, 0),
        )

    focal_score = _score_at(0.0, 0.0, 0)
    s_a = _score_at(delta_d_seg_a, delta_d_pose_a, delta_bytes_a)
    s_b = _score_at(delta_d_seg_b, delta_d_pose_b, delta_bytes_b)
    s_c = _score_at(delta_d_seg_c, delta_d_pose_c, delta_bytes_c)
    s_ab = _score_at(
        delta_d_seg_a + delta_d_seg_b,
        delta_d_pose_a + delta_d_pose_b,
        delta_bytes_a + delta_bytes_b,
    )
    s_ac = _score_at(
        delta_d_seg_a + delta_d_seg_c,
        delta_d_pose_a + delta_d_pose_c,
        delta_bytes_a + delta_bytes_c,
    )
    s_bc = _score_at(
        delta_d_seg_b + delta_d_seg_c,
        delta_d_pose_b + delta_d_pose_c,
        delta_bytes_b + delta_bytes_c,
    )
    s_abc = _score_at(
        delta_d_seg_a + delta_d_seg_b + delta_d_seg_c,
        delta_d_pose_a + delta_d_pose_b + delta_d_pose_c,
        delta_bytes_a + delta_bytes_b + delta_bytes_c,
    )

    g_a = focal_score - s_a
    g_b = focal_score - s_b
    g_c = focal_score - s_c
    sum_individual = g_a + g_b + g_c
    triple_gain = focal_score - s_abc

    # Pairwise pose-pose corrections (only nonzero if both pose deltas > 0)
    def _pair_pose_correction(p_da: float, p_db: float) -> float:
        p = focal_d_pose
        pa = max(p - p_da, 0.0)
        pb = max(p - p_db, 0.0)
        pab = max(p - p_da - p_db, 0.0)
        return -(
            math.sqrt(POSE_COEFFICIENT_INSIDE_SQRT * p)
            - math.sqrt(POSE_COEFFICIENT_INSIDE_SQRT * pa)
            - math.sqrt(POSE_COEFFICIENT_INSIDE_SQRT * pb)
            + math.sqrt(POSE_COEFFICIENT_INSIDE_SQRT * pab)
        )

    corr_ab = _pair_pose_correction(delta_d_pose_a, delta_d_pose_b)
    corr_ac = _pair_pose_correction(delta_d_pose_a, delta_d_pose_c)
    corr_bc = _pair_pose_correction(delta_d_pose_b, delta_d_pose_c)

    # Triple correction: residual after individual + pairwise
    pairwise_total = corr_ab + corr_ac + corr_bc
    triple_correction = triple_gain - sum_individual - pairwise_total

    nominal_ratio = (
        triple_gain / sum_individual if sum_individual > 0 else 1.0
    )

    notes: list[str] = []
    pose_count = sum(
        1 for d in (delta_d_pose_a, delta_d_pose_b, delta_d_pose_c) if d > 0
    )
    if pose_count == 3:
        notes.append(
            f"All three lanes touch pose. Triple Volterra correction = "
            f"{triple_correction:+.6f} (positive = super-additive at the "
            f"third order; pose compounds further as we approach Shannon floor)."
        )
    elif pose_count == 2:
        notes.append(
            "Two lanes touch pose; one disjoint. Pairwise pose correction is "
            "non-zero; triple correction should be ~0 since the third lane is "
            "linear."
        )
    elif pose_count <= 1:
        notes.append(
            "At most one pose lane; stacking is purely additive at every order."
        )

    return TripleStackingPrediction(
        focal_d_seg=focal_d_seg,
        focal_d_pose=focal_d_pose,
        focal_archive_bytes=focal_archive_bytes,
        focal_score=focal_score,
        delta_d_pose_a=delta_d_pose_a,
        delta_d_pose_b=delta_d_pose_b,
        delta_d_pose_c=delta_d_pose_c,
        delta_d_seg_a=delta_d_seg_a,
        delta_d_seg_b=delta_d_seg_b,
        delta_d_seg_c=delta_d_seg_c,
        delta_bytes_a=delta_bytes_a,
        delta_bytes_b=delta_bytes_b,
        delta_bytes_c=delta_bytes_c,
        score_after_a_only=s_a,
        score_after_b_only=s_b,
        score_after_c_only=s_c,
        score_after_ab=s_ab,
        score_after_ac=s_ac,
        score_after_bc=s_bc,
        score_after_abc=s_abc,
        individual_gain_a=g_a,
        individual_gain_b=g_b,
        individual_gain_c=g_c,
        sum_individual_gains=sum_individual,
        stacked_gain=triple_gain,
        pairwise_pose_corrections={
            "AB": corr_ab,
            "AC": corr_ac,
            "BC": corr_bc,
        },
        triple_pose_correction=triple_correction,
        nominal_stack_ratio=nominal_ratio,
        notes=notes,
    )


__all__ = [
    "StackingPrediction",
    "TripleStackingPrediction",
    "predict_stacking",
    "predict_triple_stacking",
]
