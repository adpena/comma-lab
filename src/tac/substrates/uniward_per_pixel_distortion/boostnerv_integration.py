# SPDX-License-Identifier: MIT
"""UNIWARD per-pixel routing integration INTO BoostNeRV-PR110-residual loss path.

Sister of `score_aware_loss.compose_uniward_weighted_score_loss` (free-RGB-tensor
loss extension; tested in N+1 sister `3316721639` → PARADIGM-NULL-NO-EFFECT
because the free-RGB-tensor architecture has no parameter bottleneck). THIS
module integrates UNIWARD per-pixel weighting INTO BoostNeRV-PR110-residual's
**capacity-constrained** trained-loss-path where the ~3K-params/round
ResidualHeadMLX bottleneck enables UNIWARD routing to have structural traction.

Per N+1 verdict DIAGNOSED mechanism: UNIWARD per-pixel weight map IS VALID
(19.71x dynamic range; non-degenerate real-scorer gradients) but routing
requires a parameter bottleneck the reweighting can redirect against. The
BoostNeRV-PR110-residual ResidualHeadMLX (residual_hidden_dim=12, two convs)
provides ~0.24 params/pixel at 96x128 grid — substantially bottlenecked vs
free-RGB N+1 ~36,864 params/pair architecture.

Per Catalog #230 sister-disjoint discipline: BoostNeRV substrate is READ-ONLY
consumer-imported via `from tac.substrates.boost_nerv_pr110_residual import ...`;
this module does NOT modify BoostNeRV substrate's training/test paths.

Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD: UNIWARD per-pixel routing is the
unique-per-method optimization layered ON BoostNeRV's canonical substrate-
engineering foundation (frozen-base + residual + iterative boosting).

Per CLAUDE.md "MLX portable-local-substrate authority": training-time only;
output tagged `[macOS-MLX research-signal]` per Catalog #192/#317/#341.

Entropy-position P2 loss-shape (TRAIN phase BEFORE entropy coder); MLX-first
training; numpy-portable inflate (compress-only weighting; weight map NOT
shipped per Carmack-preferred budget conservation per HNeRV parity L4).

Per Catalog #323 canonical Provenance: every loss factory return value carries
score_claim=False + promotable=False + axis_tag=[predicted] non-promotable
markers. Per Catalog #341: routing recommendation IS NOT a score signal.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

# Mirror canonical contest formula constants from CLAUDE.md non-negotiable
# (avoid circular import; values are formula constants the contest scorer
# applies at evaluate.py)
CONTEST_RATE_WEIGHT = 25.0
CONTEST_SEG_WEIGHT = 100.0
CONTEST_POSE_SQRT_INNER = 10.0
CONTEST_RATE_DENOM_BYTES = 37_545_489

# UNIWARD coefficient SAFE band (sister of score_aware_loss default)
DEFAULT_UNIWARD_LAMBDA_BOOSTNERV = 0.01

__all__ = [
    "INTEGRATION_NAME",
    "INTEGRATION_VERSION",
    "compose_uniward_into_boostnerv_loss",
    "apply_per_pixel_weight_to_residual_error",
    "build_canonical_provenance_for_integration",
]

INTEGRATION_NAME = "uniward_per_pixel_into_boostnerv_pr110_residual"
INTEGRATION_VERSION = "v1_2026-05-26"


def build_canonical_provenance_for_integration(
    *,
    uniward_lambda: float,
    weight_map_dynamic_range: float,
    pair_index: int | None = None,
) -> dict:
    """Build canonical Provenance per Catalog #323 for an integration loss call.

    Returns canonical non-promotable markers per Catalog #341 + Catalog #317
    so this loss factory's outputs cannot leak into score/promotion signals.
    """
    return {
        "integration_id": INTEGRATION_NAME,
        "integration_version": INTEGRATION_VERSION,
        "uniward_lambda": float(uniward_lambda),
        "weight_map_dynamic_range_ratio": float(weight_map_dynamic_range),
        "consumed_substrate_id": "boost_nerv_pr110_residual",
        "consumed_substrate_scope": "read_only_consumer_import",
        "pair_index": int(pair_index) if pair_index is not None else None,
        # Canonical Provenance non-promotable markers per Catalog #341
        "evidence_grade": "macOS-MLX research-signal",
        "score_claim": False,
        "promotable": False,
        "axis_tag": "[predicted]",
        "hardware_substrate_recommendation": "darwin_arm64_m5_max_mlx_local",
        "measurement_axis": "[macOS-MLX research-signal]",
        # Sister hooks per Catalog #125
        "hook_numbers_fired": [1, 5],  # sensitivity-map + continual-learning
        "entropy_position": "P2_loss_shape_TRAIN_phase",
        # Sister-disjoint discipline acknowledgment per Catalog #230
        "boostnerv_substrate_modification_scope": "none_read_only_consumer_import",
    }


def apply_per_pixel_weight_to_residual_error(
    *,
    residual_error_per_pixel: np.ndarray,
    weight_map_per_pixel: np.ndarray,
) -> np.ndarray:
    """Apply UNIWARD per-pixel weight map to per-pixel residual error.

    Parameters
    ----------
    residual_error_per_pixel : np.ndarray, shape (H, W) or (B, H, W)
        Per-pixel reconstruction error magnitude (e.g. (rgb_composed - gt).abs()
        averaged over channels OR (rgb_composed - gt)**2 averaged over channels).
        Per-pixel form REQUIRED so UNIWARD routing operates at per-pixel
        granularity (not scalar-loss).
    weight_map_per_pixel : np.ndarray, shape (H, W)
        UNIWARD weight map from `weight_map.compute_per_pixel_uniward_weight_map_numpy`.
        Higher weight = lower scorer sensitivity = safe to perturb.

    Returns
    -------
    np.ndarray, same shape as residual_error_per_pixel
        Element-wise weighted residual error.

    Notes
    -----
    Per Fridrich UNIWARD inverse-steganalysis canonical: weight is INVERSE Fisher-
    info; multiplying per-pixel error BY weight means low-sensitivity zones
    (high weight) ABSORB more error before contributing to loss → gradient
    routes perturbation TO low-sensitivity zones during BoostNeRV residual
    training.

    This shape mirrors `score_aware_loss.compose_uniward_weighted_score_loss`'s
    per-pixel weighting semantics but operates on the BoostNeRV residual-error
    surface specifically.
    """
    if residual_error_per_pixel.ndim == 2:
        if residual_error_per_pixel.shape != weight_map_per_pixel.shape:
            raise ValueError(
                f"shape mismatch: residual_error={residual_error_per_pixel.shape} vs "
                f"weight_map={weight_map_per_pixel.shape}"
            )
        return residual_error_per_pixel * weight_map_per_pixel
    elif residual_error_per_pixel.ndim == 3:
        # (B, H, W) — broadcast weight_map across batch
        if residual_error_per_pixel.shape[1:] != weight_map_per_pixel.shape:
            raise ValueError(
                f"shape mismatch: residual_error={residual_error_per_pixel.shape} vs "
                f"weight_map={weight_map_per_pixel.shape} (must match on (H, W) axes)"
            )
        return residual_error_per_pixel * weight_map_per_pixel[None, :, :]
    else:
        raise ValueError(
            f"residual_error_per_pixel must be 2D (H,W) or 3D (B,H,W); got "
            f"shape={residual_error_per_pixel.shape}"
        )


def compose_uniward_into_boostnerv_loss(
    *,
    boostnerv_residual_error_per_pixel: np.ndarray,
    weight_map_per_pixel: np.ndarray,
    scorer_loss_components: dict[str, float] | None = None,
    uniward_lambda: float = DEFAULT_UNIWARD_LAMBDA_BOOSTNERV,
    pair_index: int | None = None,
) -> dict:
    """Compose UNIWARD per-pixel weighting INTO BoostNeRV residual loss path.

    Factory pattern: returns a dict with composed loss components + canonical
    Provenance per Catalog #323. The composed total_loss can be backpropagated
    through MLX/torch as the BoostNeRV-residual trainer's primary objective.

    Parameters
    ----------
    boostnerv_residual_error_per_pixel : np.ndarray, shape (H, W) or (B, H, W)
        Per-pixel residual error from `compose_pr110_base_plus_residual(...)`
        output minus GT, magnitude-reduced over channels. This IS the surface
        UNIWARD routing reweights against — the capacity-bottleneck signal
        the N+1 verdict diagnosed as the missing element.
    weight_map_per_pixel : np.ndarray, shape (H, W)
        UNIWARD per-pixel weight map (Fisher-info inverse from BOTH scorers).
        Reuse cached gradients from sister N+1 `real_scorer_gradients_cache.npz`
        for apples-to-apples comparison.
    scorer_loss_components : dict, optional
        Canonical scorer loss components (`seg_distortion`, `pose_distortion`)
        from `score_pair_components_with_cache(...)` per Catalog #164/#226.
        If provided, composes scorer-loss + UNIWARD-weighted residual term.
        If None, returns UNIWARD-weighted-residual-only (pure-distortion attack).
    uniward_lambda : float
        Coefficient on UNIWARD-weighted residual term. SAFE band [0.0, 0.05].
    pair_index : int, optional
        Per-pair index for canonical Provenance routing.

    Returns
    -------
    dict
        Composed loss components:
        - ``uniward_weighted_residual_mean``: scalar mean of weighted per-pixel residual
        - ``score_loss_seg`` (if scorer_loss_components provided): canonical SegNet term
        - ``score_loss_pose`` (if scorer_loss_components provided): canonical PoseNet term
        - ``total_loss``: weighted composition
        - ``provenance``: canonical Provenance dict per Catalog #323
        - ``per_pixel_weighted_error``: full per-pixel array for downstream observability

    Notes
    -----
    Per Catalog #341 routing markers: provenance.score_claim=False +
    promotable=False + axis_tag=[predicted]; this loss is a TRAINING SIGNAL,
    NOT a contest score measurement.

    Per CLAUDE.md "Apples-to-apples evidence discipline": authoritative scoring
    requires exact archive bytes through `upstream/evaluate.py` on
    contest-compliant 1:1 hardware. This factory is the TRAINING-time signal
    shaping the BoostNeRV residual learner during compress phase.
    """
    if uniward_lambda < 0.0 or uniward_lambda > 0.1:
        raise ValueError(
            f"uniward_lambda must be in SAFE band [0.0, 0.1]; got {uniward_lambda!r}"
        )

    weighted_residual = apply_per_pixel_weight_to_residual_error(
        residual_error_per_pixel=boostnerv_residual_error_per_pixel,
        weight_map_per_pixel=weight_map_per_pixel,
    )
    # Mean over all axes → scalar
    weighted_residual_mean = float(weighted_residual.mean())
    weight_map_dynamic_range = float(
        weight_map_per_pixel.max() / max(weight_map_per_pixel.min(), 1e-12)
    )

    uniward_term = uniward_lambda * weighted_residual_mean

    result = {
        "uniward_weighted_residual_mean": weighted_residual_mean,
        "uniward_term": uniward_term,
        "per_pixel_weighted_error": weighted_residual,
    }

    if scorer_loss_components is not None:
        if "seg_distortion" not in scorer_loss_components:
            raise ValueError(
                "scorer_loss_components must include 'seg_distortion' if provided"
            )
        if "pose_distortion" not in scorer_loss_components:
            raise ValueError(
                "scorer_loss_components must include 'pose_distortion' if provided"
            )
        seg_d = float(scorer_loss_components["seg_distortion"])
        pose_d = float(scorer_loss_components["pose_distortion"])
        score_loss_seg = CONTEST_SEG_WEIGHT * seg_d
        score_loss_pose = math.sqrt(
            CONTEST_POSE_SQRT_INNER * max(pose_d, 0.0)
        )
        total_loss = score_loss_seg + score_loss_pose + uniward_term
        result["score_loss_seg"] = score_loss_seg
        result["score_loss_pose"] = score_loss_pose
        result["total_loss"] = total_loss
    else:
        result["total_loss"] = uniward_term

    result["provenance"] = build_canonical_provenance_for_integration(
        uniward_lambda=uniward_lambda,
        weight_map_dynamic_range=weight_map_dynamic_range,
        pair_index=pair_index,
    )
    return result
