# SPDX-License-Identifier: MIT
"""UNIWARD-weighted score-aware loss extension (per-pixel sensitivity weighting).

Composes the canonical scorer-preprocess routing (per Catalog #164/#226 via
`tac.substrates.score_aware_common.score_pair_components`) with a per-pixel
UNIWARD weight map per Fridrich-canonical inverse-steganalysis.

Sister score-aware losses (e.g. boost_nerv_pr110_residual) use scalar-weighted
contest scoring. THIS substrate extends with per-pixel weighting:

    loss = canonical_score_loss + lambda * sum(perturbation_map * weight_map)

where higher weight = lower scorer sensitivity = safe to perturb, so the
perturbation term FAVORS shifting reconstruction error into low-sensitivity
zones.

Per CLAUDE.md "MLX portable-local-substrate authority": training-time only;
output tagged `[macOS-MLX research-signal]` per Catalog #192/#317.

Entropy-position P2 loss-shape per just-elevated entropy-position discipline.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

# Mirror canonical contest weights from tac.substrates.score_aware_common
# (avoid circular import; values are CLAUDE.md non-negotiable formula
# constants per the contest scoring function)
CONTEST_RATE_WEIGHT = 25.0
CONTEST_SEG_WEIGHT = 100.0
CONTEST_POSE_SQRT_WEIGHT = math.sqrt(10.0)
CONTEST_RATE_DENOM_BYTES = 37_545_489

# UNIWARD perturbation cost coefficient — bounded SAFE band [0.0, 0.05] so the
# scorer-loss-magnitude DOMINATES the UNIWARD-perturbation-cost term during
# training (avoids the PR105 kitchen_sink anti-pattern where auxiliary terms
# overpowered primary objective).
DEFAULT_UNIWARD_LAMBDA = 0.01


def compose_uniward_weighted_score_loss(
    *,
    scorer_loss_components: dict[str, float],
    perturbation_per_pixel: np.ndarray,
    weight_map_per_pixel: np.ndarray,
    uniward_lambda: float = DEFAULT_UNIWARD_LAMBDA,
) -> dict[str, float]:
    """Compose canonical scorer loss + UNIWARD per-pixel perturbation cost term.

    Parameters
    ----------
    scorer_loss_components : dict
        Output of canonical `score_pair_components_with_cache(...)` -- expects
        keys ``seg_distortion`` and ``pose_distortion`` (both scalar floats
        already in contest-axis units).
    perturbation_per_pixel : np.ndarray, shape (H, W) or (B, H, W)
        Per-pixel reconstruction error magnitude (e.g. |pred - gt|.mean(axis=channel)).
    weight_map_per_pixel : np.ndarray, shape (H, W)
        UNIWARD weight map from `weight_map.compute_per_pixel_uniward_weight_map_numpy`.
        Higher weight = lower scorer sensitivity = safe to perturb.
    uniward_lambda : float
        Coefficient on the UNIWARD perturbation-cost term. Default 0.01 (SAFE
        band; bounded so scorer loss dominates).

    Returns
    -------
    dict
        Loss components with canonical Provenance per Catalog #323:
        - ``score_loss_seg``: canonical SegNet term (100 * d_seg)
        - ``score_loss_pose``: canonical PoseNet term (sqrt(10 * d_pose))
        - ``uniward_perturbation_cost``: per-pixel weighted perturbation sum
        - ``total_loss``: weighted composition
        - ``provenance``: dict with substrate id, hooks fired, evidence grade

    Notes
    -----
    The UNIWARD perturbation cost is INVERSE-WEIGHTED: low-sensitivity zones
    (high weight_map) contribute MORE perturbation budget per unit of cost,
    so the gradient ENCOURAGES routing perturbation TO low-sensitivity zones.

    Per CLAUDE.md "Apples-to-apples evidence discipline": this loss is a
    TRAINING SIGNAL, not a score measurement. Authoritative scoring requires
    exact archive bytes through `upstream/evaluate.py`.
    """
    if "seg_distortion" not in scorer_loss_components:
        raise ValueError("scorer_loss_components must include 'seg_distortion'")
    if "pose_distortion" not in scorer_loss_components:
        raise ValueError("scorer_loss_components must include 'pose_distortion'")

    seg_distortion = float(scorer_loss_components["seg_distortion"])
    pose_distortion = float(scorer_loss_components["pose_distortion"])

    score_loss_seg = CONTEST_SEG_WEIGHT * seg_distortion
    score_loss_pose = CONTEST_POSE_SQRT_WEIGHT * math.sqrt(max(pose_distortion, 0.0))

    # UNIWARD perturbation cost: per-pixel weighted sum
    # Inverse-weighting: low-sensitivity (high weight) absorbs more perturbation
    if perturbation_per_pixel.ndim == 3:
        perturbation_2d = perturbation_per_pixel.mean(axis=0)  # average over batch
    else:
        perturbation_2d = perturbation_per_pixel
    if perturbation_2d.shape != weight_map_per_pixel.shape:
        raise ValueError(
            f"shape mismatch: perturbation={perturbation_2d.shape} vs "
            f"weight_map={weight_map_per_pixel.shape}"
        )
    # Note: per Fridrich UNIWARD, weight is INVERSE Fisher-info, so we
    # MULTIPLY perturbation by weight (high weight = safe zone = let
    # perturbation grow). The total cost is the perturbation-to-weight
    # ratio summed: low ratio means perturbation concentrated in safe zones.
    # We log the un-normalized sum for observability + apply mean for
    # loss-magnitude stability across resolutions.
    weighted_perturbation = (perturbation_2d * weight_map_per_pixel).mean()
    uniward_perturbation_cost = float(uniward_lambda * weighted_perturbation)

    total_loss = score_loss_seg + score_loss_pose + uniward_perturbation_cost

    return {
        "score_loss_seg": score_loss_seg,
        "score_loss_pose": score_loss_pose,
        "uniward_perturbation_cost": uniward_perturbation_cost,
        "total_loss": total_loss,
        "provenance": {
            "substrate_id": "uniward_per_pixel_distortion",
            "substrate_version": "v1_2026-05-26",
            "evidence_grade": "macOS-MLX research-signal",
            "score_claim": False,
            "promotable": False,
            "axis_tag": "[predicted]",
            "hook_numbers_fired": [1, 5],
            "entropy_position": "P2_loss_shape_TRAIN_phase",
        },
    }
